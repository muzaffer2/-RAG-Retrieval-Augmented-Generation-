import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
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
def create_vector_store(documents):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.from_texts(documents, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Vektör veritabanı oluşturulurken hata oluştu: {e}")
        return None

# --- Direkt Gemini API ile Cevap Üretme ---
def get_answer(context, question, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Sana verilen bağlamı kullanarak soruyu olabildiğince detaylı bir şekilde Türkçe yanıtla. 
        Eğer cevabı bağlamda bulamazsan, "Üzgünüm, bu bilgiye sahip değilim." de. Kendi bilgini kullanma.

        Bağlam:
        {context}

        Soru: {question}

        Cevap:
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Hata: {str(e)}"

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
        with st.spinner("Vektör veritabanı oluşturuluyor... (İlk seferde biraz zaman alabilir)"):
            vector_store = create_vector_store(documents)
        
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
                            
                            # Bağlamı oluştur
                            context = "\n\n".join([doc.page_content for doc in docs])
                            
                            # Cevap al
                            answer = get_answer(context, user_question, api_key)
                            
                            st.write("### 💬 Cevap:")
                            st.write(answer)
                            
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
