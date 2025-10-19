import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    metadata_list = []
    df.columns = df.columns.str.strip()
    
    for index, row in df.iterrows():
        try:
            # Daha detaylı ve aranabilir text oluştur
            text = (
                f"Oyuncu: {row['Player']}\n"
                f"Takım: {row['Tm']}\n"
                f"Tarih: {row['Data']}\n"
                f"Rakip: {row['Opp']}\n"
                f"Süre: {row['MP']} dakika\n"
                f"Sayı: {row['PTS']} puan\n"
                f"Ribaund: {row['TRB']}\n"
                f"Asist: {row['AST']}\n"
                f"Top Çalma: {row['STL']}\n"
                f"Blok: {row['BLK']}\n"
                f"Saha İçi Başarı Yüzdesi: {float(row['FG%'])*100:.1f}%\n\n"
                f"Özet: {row['Player']} ({row['Tm']}), {row['Data']} tarihinde {row['Opp']} takımına karşı "
                f"{row['MP']} dakika oynadı ve {row['PTS']} sayı, {row['TRB']} ribaund, {row['AST']} asist kaydetti."
            )
            documents.append(text)
            
            # Metadata ekle
            metadata_list.append({
                'player': row['Player'],
                'team': row['Tm'],
                'date': row['Data'],
                'points': row['PTS']
            })
        except (ValueError, KeyError, TypeError) as e:
            continue
    
    return df, documents, metadata_list

# --- LangChain ve FAISS ile Vektör Veritabanı Oluşturma ---
@st.cache_resource
def create_vector_store(documents, metadata_list):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Türkçe destekli model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Metadata ile birlikte vector store oluştur
        from langchain.schema import Document
        docs = [Document(page_content=text, metadata=meta) for text, meta in zip(documents, metadata_list)]
        vector_store = FAISS.from_documents(docs, embedding=embeddings)
        
        return vector_store
    except Exception as e:
        st.error(f"Vektör veritabanı oluşturulurken hata oluştu: {e}")
        return None

# --- Gemini ve LangChain ile Cevap Üretme ---
def get_conversational_chain(api_key):
    prompt_template = """
    Sen bir NBA istatistik asistanısın. Aşağıdaki oyuncu verileri bağlamında soruları yanıtlıyorsun.
    
    ÖNEMLİ KURALLAR:
    1. Sadece verilen bağlamdaki bilgileri kullan
    2. Sayısal verileri doğru bir şekilde aktar
    3. Oyuncu isimlerini tam olarak belirt
    4. Tarih bilgilerini ekle
    5. Eğer bağlamda bilgi yoksa, açıkça "Bu bilgiye sahip değilim" de
    
    BAĞLAM:
    {context}
    
    SORU: {question}
    
    CEVAP (Türkçe ve detaylı):
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0.1,  # Daha deterministik cevaplar için düşük
        google_api_key=api_key,
        convert_system_message_to_human=True
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
    
    result = preprocess_nba_data(FILE_PATH)
    if result and len(result) == 3:
        df, documents, metadata_list = result
    else:
        df, documents, metadata_list = None, None, None

    if documents:
        st.sidebar.success(f"✅ {len(documents)} oyuncu verisi yüklendi")
        
        # Vektör veritabanını oluştur
        with st.spinner("Vektör veritabanı oluşturuluyor... (İlk seferde 1-2 dakika sürebilir)"):
            vector_store = create_vector_store(documents, metadata_list)
        
        if vector_store:
            st.success("✅ Sistem hazır! Sorularınızı sorabilirsiniz.")
            
            # Örnek sorular göster
            st.sidebar.markdown("### 💡 Örnek Sorular:")
            st.sidebar.markdown("""
            - Jayson Tatum kaç sayı attı?
            - LeBron James'in ribaund sayısı nedir?
            - En yüksek sayıyı kim attı?
            - Boston Celtics oyuncularının performansı nasıl?
            """)
            
            # Kullanıcıdan soru al
            user_question = st.text_input(
                "Sorunuzu yazın:", 
                placeholder="Örnek: 'Jayson Tatum en son maçında kaç sayı attı?'"
            )

            if st.button("🔍 Soru Sor"):
                if user_question:
                    with st.spinner("Cevap aranıyor..."):
                        try:
                            # Daha fazla belge getir (k=5)
                            docs = vector_store.similarity_search(user_question, k=5)
                            
                            # Debug: Bulunan belgeleri göster
                            st.info(f"📊 {len(docs)} ilgili kayıt bulundu")
                            
                            if not docs:
                                st.warning("⚠️ İlgili veri bulunamadı. Lütfen sorunuzu farklı şekilde sorun.")
                            else:
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
                                        st.code(doc.page_content)
                                        if doc.metadata:
                                            st.json(doc.metadata)
                                        st.divider()
                        except Exception as e:
                            st.error(f"Cevap üretilirken hata oluştu: {e}")
                            st.exception(e)
                else:
                    st.warning("⚠️ Lütfen bir soru girin.")
        else:
            st.error("Vektör veritabanı oluşturulamadı.")
    else:
        st.error("❌ Veri dosyası yüklenemedi veya işlenemedi.")
        st.info("💡 'nba_fantasy_dataset.csv' dosyasının proje klasöründe olduğundan emin olun.")
else:
    st.sidebar.warning("⚠️ Lütfen başlamak için Google API anahtarınızı girin.")
    st.info("👈 Sol taraftaki alana API anahtarınızı girin.")