import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# ---------------------------------------------------------
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ---------------------------------------------------------
def load_and_process_data(filename):
    print(f"Loading dataset: {filename}...")
    df = pd.read_csv(filename)
    
    # NaN değerleri 0 ile doldur (Seyrek veri için kritik)
    df.fillna(0, inplace=True)

    # --- Sütun Ayrıştırma Mantığı ---
    # Hedef Sütunlar (Renkler): 15L, 25L... 110b
    # Veri setindeki header yapısına göre hedef sütunları belirliyoruz
    target_cols = [col for col in df.columns if any(x in col for x in ['15L', '25L', '45L', '75L', '110L', 
                                                                     '15a', '25a', '45a', '75a', '110a', 
                                                                     '15b', '25b', '45b', '75b', '110b'])]
    
    # Metadata sütunları (Eğitime girmeyecekler)
    ignore_cols = ['SampleNo', '15Si', '45Si', '75Si', '15Sa', '45Sa', '75Sa', 'G']
    
    # Geriye kalanlar Pigmentlerdir (Input Features)
    feature_cols = [col for col in df.columns if col not in target_cols and col not in ignore_cols]
    
    print(f"Tespit Edilen Pigment Sayısı (Features): {len(feature_cols)}")
    print(f"Tespit Edilen Renk Kanalı Sayısı (Targets): {len(target_cols)}")
    
    # Input (X) ve Output (y) ayrımı
    X = df[feature_cols].values
    y = df[target_cols] # Tüm hedefler burada, ama GP tek tek eğitilir
    
    return X, y, feature_cols, target_cols

# ---------------------------------------------------------
# 2. GENETİK PROGRAMLAMA MODELİ
# ---------------------------------------------------------
def train_gp_for_single_output(X_train, y_train, feature_names, target_name):
    """
    Tek bir renk kanalı (Örn: 15L) için formül evrimleştirir.
    """
    print(f"\n--- Evrim Başlıyor: Hedef {target_name} ---")
    
    # Fonksiyon Seti: Renk fiziğinde (Kubelka-Munk) log ve exp önemlidir.
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv']
    
    est_gp = SymbolicRegressor(
        population_size=1000,      # Popülasyon büyüklüğü (GPU yoksa 1000-5000 iyidir)
        generations=20,            # Kaç nesil evrimleşecek
        tournament_size=20,        # Seçilim baskısı
        stopping_criteria=0.01,    # Hedef hata (MSE)
        p_crossover=0.7,           # Çaprazlama oranı
        p_subtree_mutation=0.1,    # Mutasyon oranı
        p_hoist_mutation=0.05,     # Bloat (şişme) engelleme mutasyonu
        p_point_mutation=0.1,
        max_samples=0.9,           # Her nesilde verinin %90'ını kullan
        verbose=1,                 # Eğitim sırasında çıktı ver
        parsimony_coefficient=0.001, # Karmaşık formülleri cezalandır (Occam's Razor)
        random_state=42,
        function_set=function_set,
        feature_names=feature_names # Formülde X1, X2 yerine pigment adları yazsın
    )
    
    t0 = time.time()
    est_gp.fit(X_train, y_train)
    print(f"Eğitim Süresi: {time.time() - t0:.2f} saniye")
    
    return est_gp

# ---------------------------------------------------------
# 3. ANA UYGULAMA
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Veriyi Yükle
    filename = 'eval_datasetA.csv' # Dosya adını buraya gir
    
    try:
        X, y_all, feature_names, target_names = load_and_process_data(filename)
        
        # Veriyi Ölçekle (Pigmentler 0-1 arasına sıkıştırılır, GP daha iyi çalışır)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Eğitim/Test Ayrımı (%80 Eğitim, %20 Test)
        X_train, X_test, y_train_all, y_test_all = train_test_split(
            X_scaled, y_all, test_size=0.2, random_state=42
        )
        
        # --- DEMO: Sadece '15L' (Face Tone Lightness) değerini tahmin edelim ---
        # Projede bunu bir döngüye alıp tüm 15 kanal için yapacaksın.
        target_to_predict = '15L' 
        
        if target_to_predict in target_names:
            y_train_single = y_train_all[target_to_predict].values
            y_test_single = y_test_all[target_to_predict].values
            
            # GP Modelini Eğit
            gp_model = train_gp_for_single_output(X_train, y_train_single, feature_names, target_to_predict)
            
            # Sonuçları Değerlendir
            y_pred = gp_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test_single, y_pred))
            
            print(f"\n--- SONUÇLAR: {target_to_predict} ---")
            print(f"Test RMSE Hatası: {rmse:.4f}")
            print(f"R2 Skoru: {gp_model.score(X_test, y_test_single):.4f}")
            
            print("\n--- KEŞFEDİLEN FİZİKSEL FORMÜL ---")
            print(gp_model._program)
            
            # Formülün ne kadar karmaşık olduğunu görelim
            print(f"Formül Derinliği: {gp_model._program.depth_}")
            print(f"Formül Uzunluğu: {gp_model._program.length_}")
            
        else:
            print(f"Hata: {target_to_predict} sütunu bulunamadı.")
            
    except FileNotFoundError:
        print(f"Hata: '{filename}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")