import streamlit as st
from rag_engine import ask
from ingest import ingest
from pathlib import Path

st.set_page_config(page_title='Footnote', page_icon='📟')
st.title("📟 Footnote")
st.caption("Locally Run RAG-based Study Assistant")

#Upload area
st.subheader("Upload Notes (copyable pdfs only) : ")
uploaded_files = st.file_uploader(
  "Click to Upload",
  type=["pdf"],
  accept_multiple_files=True
)

if uploaded_files:
  Path("data").mkdir(exist_ok=True)

  for file in uploaded_files:
    with open(Path("data") / file.name, "wb") as f:
      f.write(file.getbuffer())

  if st.button("Ingest uploaded files"):
    with st.spinner("Adding uploade pdfs into my mind"):
      pdfscount, chunkscount = ingest()

    st.success(f"Ingested {pdfscount} PDFs into {chunkscount} chunks !")
    st.rerun()

#Query area
user_question = st.text_input("Ask Your Notes :")

if user_question:
  with st.spinner("Turning my gears..."):
    answer, context = ask(user_question)
  
  st.markdown('### Answer')
  st.write(answer)

  with st.expander("context used :"):
    st.write(context)
