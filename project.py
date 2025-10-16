# import torch
# print(torch.__version__)
# # 1. CUDA (GPU) kullanılabilirliğini kontrol et
# gpu_kullanilabilir = torch.cuda.is_available()
# print(f"CUDA (GPU) Kullanılabilir: {gpu_kullanilabilir}")

# if gpu_kullanilabilir:
#     # 2. Kullanılan GPU sayısını ve adını al
#     gpu_sayisi = torch.cuda.device_count()
#     gpu_adi = torch.cuda.get_device_name(0) # İlk GPU'nun adını alır
#     print(f"Kullanılan GPU Sayısı: {gpu_sayisi}")
#     print(f"GPU Adı: {gpu_adi}")

#     # 3. Basit bir tensör işlemini GPU'ya taşıyarak test et
#     try:
#         x = torch.rand(5, 5).cuda()
#         y = torch.rand(5, 5).cuda()
#         z = x + y
#         print("GPU'da basit tensör işlemi başarılı.")
#     except Exception as e:
#         print(f"GPU işlemi başarısız: {e}")
# else:
#     print("GPU kullanılamıyor, işlem CPU'da yürütülecektir.")

import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# --- Veri İşleme Fonksiyonu ---
def preprocess_nba_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=';')
    except FileNotFoundError:
        st.error(f"Hata: '{file_path}' dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
        return None, None
    except Exception as e:
        st.error(f"Dosya okunurken bir hata oluştu: {e}")
        return None, None

    documents = []
    df.columns = df.columns.str.strip()
    
    for index, row in df.iterrows():
        try:
            text = f"{row['Player']} ({row['Tm']}), {row['Data']} tarihinde {row['Opp']} takımına karşı oynanan maçta " \
                   f"{row['MP']} dakika süre aldı. Maçı {row['PTS']} sayı, {row['TRB']} ribaund, " \
                   f"{row['AST']} asist, {row['STL']} top çalma ve {row['BLK']} blok ile tamamladı. " \
                   f"Saha içi isabet oranı %{float(row['FG%'])*100:.1f} idi."
            documents.append(text)
        except (ValueError, KeyError, TypeError) as e:
            continue
    return df, documents

# --- LangChain ve FAISS ile Vektör Veritabanı Oluşturma ---
@st.cache_resource
def create_vector_store(documents, _api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=_api_key
        )
        vector_store = FAISS.from_texts(documents, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Vektör veritabanı oluşturulurken hata oluştu: {e}")
        return None

# --- Gemini ve LangChain ile Cevap Üretme ---
def get_conversational_chain(api_key):
    prompt_template = """
    Sana verilen bağlamı kullanarak soruyu olabildiğince detaylı bir şekilde Türkçe yanıtla. 
    Eğer cevabı bağlamda bulamazsan, "Üzgünüm, bu bilgiye sahip değilim." de. Kendi bilgini kullanma.\n\n
    Bağlam:\n {context}?\n
    Soru: \n{question}\n

    Cevap:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        temperature=0.3,
        google_api_key=api_key
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- Streamlit Arayüzü ---
st.set_page_config(page_title="🏀 NBA Fantezi Asistanı", layout="wide")
st.title("🏀 NBA Fantezi Asistanı Chatbot'u")
st.write("Oyuncu istatistikleri hakkında sorularınızı sorun!")

# API Anahtarını kullanıcıdan al
api_key = st.sidebar.text_input("Google API Anahtarınızı Girin:", type="password")

if api_key:
    # Veriyi yükle ve işle
    FILE_PATH = 'nba_fantasy_dataset.csv'
    df, documents = preprocess_nba_data(FILE_PATH)

    if documents:
        # Vektör veritabanını oluştur
        with st.spinner("Vektör veritabanı oluşturuluyor..."):
            vector_store = create_vector_store(documents, api_key)
        
        if vector_store:
            st.success("✅ Sistem hazır! Sorularınızı sorabilirsiniz.")
            
            # Kullanıcıdan soru al
            user_question = st.text_input("Sorunuzu yazın:", placeholder="Örnek: 'Jayson Tatum en son maçında kaç sayı attı?'")

            if st.button("🔍 Soru Sor"):
                if user_question:
                    with st.spinner("Cevap aranıyor..."):
                        try:
                            # Soruyu yanıtlama
                            docs = vector_store.similarity_search(user_question, k=3)
                            chain = get_conversational_chain(api_key)
                            response = chain(
                                {"input_documents": docs, "question": user_question}, 
                                return_only_outputs=True
                            )
                            
                            st.write("### 💬 Cevap:")
                            st.write(response["output_text"])
                            
                            # Kaynak göster
                            with st.expander("📚 Kullanılan Kaynaklar"):
                                for i, doc in enumerate(docs, 1):
                                    st.write(f"**Kaynak {i}:**")
                                    st.write(doc.page_content)
                                    st.divider()
                        except Exception as e:
                            st.error(f"Cevap üretilirken hata oluştu: {e}")
                else:
                    st.warning("⚠️ Lütfen bir soru girin.")
        else:
            st.error("Vektör veritabanı oluşturulamadı.")
    else:
        st.error("Veri dosyası yüklenemedi veya işlenemedi.")
else:
    st.sidebar.warning("⚠️ Lütfen başlamak için Google API anahtarınızı girin.")
    st.info("👈 Sol taraftaki alana API anahtarınızı girin.")
