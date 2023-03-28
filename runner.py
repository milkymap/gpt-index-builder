import re 

import zmq 
import zmq.asyncio as azmq
from io import BytesIO
import subprocess

import numpy as np 
from os import path 

import zmq 
import httpx 
import logging 

import numpy as np 

import click 
import asyncio 

from typing import List, Optional
from glob import glob 

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    filters, 
    Application,
    ContextTypes,
    CommandHandler,
    ConversationHandler,
    MessageHandler
)
from os import path 

import operator as op 

from typing import List, Tuple, Dict, Optional, Union

from strategies import load_transformer_model, get_embedding, parse_url, find_top_k, gpt3_search
import logging 

logging.basicConfig(
    format='%(asctime)s : (%(name)s) | %(filename)s -- %(lineno)3d -- %(levelname)7s -- %(message)s',
    level=logging.INFO 
)

logger = logging.getLogger(name='TelegramBot')

class GPTRunner:
    def __init__(self, token:str, name:str, description:str, transformer_model_name:str, cache_folder:str, device:str, chunks:List[str], corpus_embeddings:np.ndarray, top_k:int=7):
        self.token = token 
        self.chunks = chunks
        self.top_k = top_k

        self.name = name 
        self.description = description

        self.corpus_embeddings = corpus_embeddings
        self.transformer = load_transformer_model(
            model_name=transformer_model_name,
            cache_folder=cache_folder
        )
        self.app = Application.builder().token(token).build()
        self.device = device

    async def start(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        logger.debug(f'{user.first_name} is connected')
        await update.message.reply_text(text=f'Hello {user.first_name}, je suis le chatbot {self.name}. Voici la description de mon domaine de compétence : {self.description}')
        return 0 
    
    async def chatting(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        text = update.message.text
        query_embedding = get_embedding(text, model=self.transformer, device=self.device)
        selected_scores_indices = find_top_k(query_embedding, self.corpus_embeddings, self.top_k)
        
        chunks_acc = []
        for _, index in selected_scores_indices:
            chunk = self.chunks[index]
            chunks_acc.append(chunk)
        corpus_context = '\n'.join(chunks_acc)
        try:

            logger.debug('Searching...')

            response = gpt3_search(
                corpus_context=corpus_context,
                query=text,
                name=self.name,
                description=self.description
            )

            await update.message.reply_text(text=response) 
            return 0

        except Exception as e:
            logger.exception("Exception occurred")
            logger.error(f"Exception: {e}")

        await update.message.reply_text('Une erreur interne est survenue, merci de refaire votre demande')
        return 0 

    async def stop(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        await update.message.reply_text(
            text="""
                Merci de votre visite et n'hésitez pas à revenir si vous avez d'autres questions ou besoins d'assistance.À bientôt !
            """
        )
        return ConversationHandler.END

    def listen(self):
        self.app.run_polling()

    def __enter__(self):
        try:
            self.ctx = azmq.Context()
            
            handler = ConversationHandler(
                entry_points=[CommandHandler('start', self.start)],
                states={
                    0: [CommandHandler('stop', self.stop), MessageHandler(filters.TEXT|filters.VOICE, self.chatting)],
                },
                fallbacks=[CommandHandler('stop', self.stop)]
            )
            self.app.add_handler(handler)
        except Exception as e:
            logger.error(e)
        return self 

    def __exit__(self, exc, val, traceback):
        if exc is not None:
            logger.exception(traceback)
        self.ctx.term()
        logger.debug('GPTRunner ... shutdown')

