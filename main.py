import os 
import click 

import openai 
import wikipedia

import torch as th 

import pickle 

import tiktoken

from rich.console import Console

from log import logger 
from strategies import convert_pdf2text, split_into_chunks, chunk_embeddings, gpt3_search, find_top_k, parse_url, load_transformer_model, get_embedding

from runner import GPTRunner

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

@click.group(chain=False, invoke_without_command=True)
@click.option('--openai_api_key', type=str, required=True, envvar='OPENAI_API_KEY')
@click.option('--cache_folder', type=str, required=True, envvar='TRANSFORMERS_CACHE')
@click.option('--transformer_model_name', type=str, default='Sahajtomar/french-semantic')
@click.pass_context
def group(ctx:click.core.Context, openai_api_key:str, cache_folder:str, transformer_model_name:str):
    ctx.ensure_object(dict)
    ctx.obj['openai_api_key'] = openai_api_key
    ctx.obj['transformer_model_name'] = transformer_model_name

    ctx.obj['cache_folder'] = cache_folder
    ctx.obj['tokenizer'] = tiktoken.encoding_for_model('gpt-3.5-turbo')

    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')
    ctx.obj['device'] = device
    
    subcommand = ctx.invoked_subcommand
    if subcommand is not None:
        logger.debug(f"Invoked subcommand: {subcommand}")

@group.command()
@click.option('--path2pdf_file', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--path2extracted_features', type=click.Path(exists=False, dir_okay=False), required=True)
@click.option('--chunk_size', type=int, default=128, help='chunk size for the tokenizer')
@click.option('--batch_size', type=int, default=8, help='batch size for the cohere api')
@click.pass_context
def build_index_from_pdf(ctx:click.core.Context, path2pdf_file:str, path2extracted_features:str, chunk_size:int, batch_size:int):
    extracted_text = convert_pdf2text(path2pdf_file)
    chunks = split_into_chunks(extracted_text, tokenizer=ctx.obj['tokenizer'], chunk_size=chunk_size)
    
    transformer_model = load_transformer_model(ctx.obj['transformer_model_name'], cache_folder=ctx.obj['cache_folder'])
    embeddings = chunk_embeddings(
        transformer_model=transformer_model,
        chunks=chunks, 
        batch_size=batch_size,
        device=ctx.obj['device']
    )

    with open(path2extracted_features, 'wb') as file_pointer:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings}, file_pointer)
    logger.info(f"Saved extracted features to: {path2extracted_features}")

@group.command()
@click.option('--wikipedia_url', type=str, help='wikipedia valid url', required=True)
@click.option('--path2extracted_features', type=click.Path(exists=False, dir_okay=False), required=True)
@click.option('--chunk_size', type=int, default=128, help='chunk size for the tokenizer')
@click.option('--batch_size', type=int, default=8, help='batch size for the cohere api')
@click.pass_context
def build_index_from_wikipedia(ctx:click.core.Context, wikipedia_url:str, path2extracted_features:str, chunk_size:int, batch_size:int):
    title, language_code = parse_url(wikipedia_url)
    wikipedia.set_lang(language_code)
    page = wikipedia.page(title)
    extracted_text = page.content
    logger.info('text was extracted from wikipedia')
    chunks = split_into_chunks(extracted_text, tokenizer=ctx.obj['tokenizer'], chunk_size=chunk_size)
    
    transformer_model = load_transformer_model(ctx.obj['transformer_model_name'], cache_folder=ctx.obj['cache_folder'])
    
    embeddings = chunk_embeddings(
        transformer_model=transformer_model,
        chunks=chunks,  
        batch_size=batch_size,
        device=ctx.obj['device']
    )

    with open(path2extracted_features, 'wb') as file_pointer:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings}, file_pointer)
    logger.info(f"Saved extracted features to: {path2extracted_features}")

@group.command()
@click.option('--name', type=str, required=True)
@click.option('--description', type=str, required=True)
@click.option('--path2extracted_features', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--top_k', type=int, default=7)
@click.pass_context
def explore_index(ctx:click.core.Context, name:str, description:str, path2extracted_features:str, top_k:int):
    
    console = Console()

    openai.api_key = ctx.obj['openai_api_key']

    with open(path2extracted_features, 'rb') as file_pointer:
        extracted_features = pickle.load(file_pointer)
    
    logger.info(f"Loaded extracted features from: {path2extracted_features}")

    transformer_model = load_transformer_model(ctx.obj['transformer_model_name'], cache_folder=ctx.obj['cache_folder'])
   
    keep_looping = True
    while keep_looping:
        try:
            query = input('USER: ')
            if query == 'exit':
                keep_looping = False
            else:
                
                logger.debug('Embedding query...')

                query_embedding = transformer_model.encode(query, device=ctx.obj['device'], show_progress_bar=False)
                selected_scores_indices = find_top_k(query_embedding, extracted_features['embeddings'], k=top_k)
                chunks_acc = []
                for _, index in selected_scores_indices:
                    chunk = extracted_features['chunks'][index]
                    chunks_acc.append(chunk)
                corpus_context = '\n'.join(chunks_acc)
                try:

                    logger.debug('Searching...')

                    response = gpt3_search(
                        corpus_context=corpus_context,
                        query=query,
                        name=name,
                        description=description
                    )
                    console.print(f"ASSISTANT : {response}", style="bold green")
                except Exception as e:
                    logger.exception("Exception occurred")
                    logger.error(f"Exception: {e}")

        except KeyboardInterrupt:
            keep_looping = False
        except Exception:
            logger.exception("Exception occurred")
            keep_looping = False

@group.command()
@click.option('--name', type=str, required=True)
@click.option('--description', type=str, required=True)
@click.option('--path2extracted_features', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--top_k', type=int, default=7)
@click.option('--telegram_token', type=str, required=True)
@click.pass_context
def deploy_on_telegram(ctx:click.core.Context, name:str, description:str, path2extracted_features:str, top_k:int, telegram_token:str):
    with open(path2extracted_features, 'rb') as file_pointer:
        extracted_features = pickle.load(file_pointer)
    
    logger.info(f"Loaded extracted features from: {path2extracted_features}")
    with GPTRunner(
        token=telegram_token,
        name=name,
        description=description,
        top_k=top_k,
        transformer_model_name=ctx.obj['transformer_model_name'],
        cache_folder=ctx.obj['cache_folder'],
        device=ctx.obj['device'],
        corpus_embeddings=extracted_features['embeddings'],
        chunks=extracted_features['chunks']
    ) as runner:
        runner.listen()


if __name__ == '__main__':
    group(obj={})