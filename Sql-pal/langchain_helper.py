import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain

load_dotenv()

def get_secret(key, default=None):
    return os.getenv(key) or st.secrets.get(key, default)


def sql_bot_with_query():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=get_secret("GOOGLE_API_KEY"),
        temperature=0.1
    )
    db_user = get_secret("DB_USER")

    db_password = get_secret("DB_PASSWORD")

    db_host = get_secret("DB_HOST")

    db_name = get_secret("DB_NAME")

    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3
    )
    query_chain = create_sql_query_chain(llm, db)
    return query_chain, db


def sql_bot_direct():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=get_secret("GOOGLE_API_KEY"),
        temperature=0.1
    )

    db_user = get_secret("DB_USER")

    db_password = get_secret("DB_PASSWORD")

    db_host = get_secret("DB_HOST")

    db_name = get_secret("DB_NAME")

    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3
    )

    return llm, db













#
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.utilities import SQLDatabase
# from langchain_experimental.sql import SQLDatabaseChain
# from langchain.chains import create_sql_query_chain
#
# load_dotenv()
#
#
# def sql_bot_with_query():
#
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         google_api_key=os.getenv("GOOGLE_API_KEY"),
#         temperature=0.1
#     )
#
#     db_user = os.getenv("DB_USER")
#     db_password = os.getenv("DB_PASSWORD")
#     db_host = os.getenv("DB_HOST")
#     db_name = os.getenv("DB_NAME")
#
#     db = SQLDatabase.from_uri(
#         f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
#         sample_rows_in_table_info=3
#     )
#     query_chain = create_sql_query_chain(llm, db)
#     return query_chain, db
#
# def sql_bot_direct():
#
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         google_api_key=os.getenv("GOOGLE_API_KEY"),
#         temperature=0.1
#     )
#
#     db_user = os.getenv("DB_USER")
#     db_password = os.getenv("DB_PASSWORD")
#     db_host = os.getenv("DB_HOST")
#     db_name = os.getenv("DB_NAME")
#
#     db = SQLDatabase.from_uri(
#         f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
#         sample_rows_in_table_info=3
#     )
#
#     return llm, db