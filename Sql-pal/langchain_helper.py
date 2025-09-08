import os
import pymysql
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

load_dotenv()

def sql_bot():

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")

    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3
    )


    chain=SQLDatabaseChain.from_llm(
        llm,
        db,
        verbose=True,
        return_direct=True
    )
    return chain

if __name__ == "__main__":
    chain = sql_bot()
    result = chain.run("How many t-shirt do we have left for Nike in extra small size and white color?")
    print("Query Result:", result)
