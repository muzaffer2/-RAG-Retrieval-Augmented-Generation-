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

import pandas as pd

def preprocess_nba_data(file_path):
    """
    NBA fantezi veri setini okur ve her satırı RAG modeli için
    doğal dil metnine dönüştürür.
    """
    try:
        # HATA DÜZELTİLDİ: Delimiter'ı noktalı virgül (';') olarak değiştiriyoruz.
        df = pd.read_csv(file_path, delimiter=';')

        # Hata ayıklama kodunu artık kaldırabiliriz veya yorum satırı yapabiliriz.
        # print("--- Tespit Edilen Sütun Adları ---")
        # print(df.columns)
        # print("---------------------------------")

    except Exception as e:
        print(f"Dosya okunurken bir hata oluştu: {e}")
        return None, None

    documents = []
    # DataFrame'deki sütun adlarında olası gizli boşlukları temizleyelim.
    df.columns = df.columns.str.strip()
    
    for index, row in df.iterrows():
        try:
            # Her satırdaki veriyi anlamlı bir cümleye dönüştürüyoruz.
            text = f"{row['Player']} ({row['Tm']}), {row['Data']} tarihinde {row['Opp']} takımına karşı oynanan maçta " \
                   f"{row['MP']} dakika süre aldı. Maçı {row['PTS']} sayı, {row['TRB']} ribaund, " \
                   f"{row['AST']} asist, {row['STL']} top çalma ve {row['BLK']} blok ile tamamladı. " \
                   f"Saha içi isabet oranı %{float(row['FG%'])*100:.1f} idi."
            documents.append(text)
        except (ValueError, KeyError) as e:
            # Olası hatalı satırları atlamak için bir kontrol ekliyoruz.
            print(f"Satır {index+1} işlenirken bir hata oluştu ve atlandı: {e}")
            continue


    return df, documents

# --- KODU ÇALIŞTIRMA ---
FILE_PATH = 'nba_fantasy_dataset.csv'
df, processed_documents = preprocess_nba_data(FILE_PATH)

if processed_documents:
    print("\n" + "="*50 + "\n")
    print("--- Veri Başarıyla İşlendi! ---")
    print("--- RAG için Hazırlanmış İlk 5 Metin Dokümanı ---")
    for doc in processed_documents[:5]:
        print(f"- {doc}\n")