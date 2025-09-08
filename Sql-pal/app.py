import streamlit as st
from langchain_helper import sql_bot_with_query
import re


st.set_page_config(page_title="SQL Chat Bot", page_icon="🤖")
st.title("SQL Chat Bot \n*Developed By Sheryar*")
st.write("Ask natural language questions about your database")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sql_query" in message:
            with st.expander("View SQL Query"):
                st.code(message["sql_query"], language="sql")



def clean_result(result):
    #cleans the result
    if isinstance(result, str):

        if result.startswith("[(") and result.endswith(")]"):

            match = re.search(r'\((\d+)\,?\)', result)
            if match:
                return match.group(1)


        decimal_match = re.search(r"Decimal\('(\d+)'\)", result)
        if decimal_match:
            return decimal_match.group(1)

        return result
    return str(result)



def clean_sql_query(sql_query):

    if sql_query.startswith("SQLQuery:"):
        sql_query = sql_query.replace("SQLQuery:", "", 1).strip()


    sql_query = re.sub(r'^```sql\s*|\s*```$', '', sql_query).strip()


    sql_query = re.sub(r'SQLQuery:\s*', '', sql_query)

    return sql_query



if prompt := st.chat_input("Ask a question about your database..."):

    st.session_state.messages.append({"role": "user", "content": prompt})


    with st.chat_message("user"):
        st.write(prompt)


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:

                query_chain, db = sql_bot_with_query()

                sql_query = query_chain.invoke({"question": prompt})
                clean_sql = clean_sql_query(sql_query)
                result = db.run(clean_sql)
                cleaned_result = clean_result(result)
                st.write(cleaned_result)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": cleaned_result,
                    "sql_query": clean_sql
                })

                with st.expander("View SQL Query"):
                    st.code(clean_sql, language="sql")

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })