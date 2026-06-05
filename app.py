import streamlit as st
import requests
from pathlib import Path

st.set_page_config(page_title='Footnote', page_icon='📟')
st.title("📟 Footnote")
st.caption("Locally Run RAG-based Study Assistant")

request_url  = "http://127.0.0.1:8000"

with st.sidebar:
  st.header("🗺️ navigation")
  current_page = st.radio("Go to:", ["💬 Ask Your Documents", "🗂️ Upload & Manage Docs"])
  
  st.divider()
  
  st.header("🧠 Index Manager")
  try:
    files_resp = requests.get(f"{request_url}/files")
    if files_resp.status_code == 200:
      active_files = files_resp.json().get("files", [])
      
      if active_files:
        st.write("Currently Ingested Files:")
        for f in active_files:
          st.code(f)
      else:
        st.info("No files ingested yet. Feed Me.")
  except requests.exceptions.ConnectionError:
    st.error("Backend offline.")
      
  st.divider()

  if st.button("🚨 Clear complete index", use_container_width=True):
    try:
      wipe_resp = requests.delete(f"{request_url}/clear")
      if wipe_resp.status_code == 200:
        st.toast("memory wiped") 
        st.rerun()
      else:
        st.error("failed to wipe memory.")
    except requests.exceptions.ConnectionError:
      st.error("backend offline.")


# ----------------------------------------
# PAGE 1: CHAT AREA
# ----------------------------------------
if current_page == "💬 Ask Your Documents":
  st.subheader("ask your files a question")
  user_question = st.chat_input("Ask Your Files :")

  if user_question:
    with st.spinner("Turning my gears..."):
      try:
        payload = {"question" : user_question}
        response = requests.post(f"{request_url}/ask", json=payload)

        if response.status_code == 200:
          data = response.json()
          answer = data["answer"]
          context = data["context"]
          st.markdown('### Answer')
          st.write(answer)
          with st.expander("context used :"):
            st.write(context)
        else :
          st.error(f"backend error with response code : {response.status_code}")

      except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI.")

# ----------------------------------------
# PAGE 2: UPLOAD AREA
# ----------------------------------------
elif current_page == "🗂️ Upload & Manage Docs":
  st.subheader("Upload Files (text-based pdfs only) : ")
  uploaded_files = st.file_uploader(
    "Click to Upload",
    type=["pdf"],
    accept_multiple_files=True
  )

  if uploaded_files:
    if(st.button("Hit this to ingest")):
      with st.status("Ingesting into FAISS Memory...") as status:
        for file in uploaded_files:
          try:
            file_bytes = file.getvalue()
            files = {"file": (file.name, file_bytes, "application/pdf")}
            status.write(f"Current file processing into FAISS : {file.name}")

            response = requests.post(f"{request_url}/ingest", files=files)
            
            if response.status_code == 200:
              result = response.json()
              st.success(f"successfully loaded! Index now contains {result['ingested no. of pdfs']} files.")
            else:
              st.error(f"failed to process document {file.name} on backend server.")
                  
          except requests.exceptions.ConnectionError:
            st.error("backend server is offline.")
        status.update(label="Ingestion Complete!", state="complete", expanded=False)
      st.success("All uploaded files succesfully processed. Ready for a question.")