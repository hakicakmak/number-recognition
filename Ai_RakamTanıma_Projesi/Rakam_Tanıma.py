import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import warnings

# Uyarıları filtrele
warnings.filterwarnings("ignore", category=UserWarning)

# Bu işlem 1-2 dk sürebilir..
mnist = fetch_openml('mnist_784', parser='auto')

# test ve train oranı 1/7 ve 6/7
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

scaler = StandardScaler()
# Scaler'ı sadece training set üzerinde fit yapmamız yeterli
scaler.fit(train_img)
# Ama transform işlemini hem training sete hem de test sete yapmamız gerekiyor..
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

pca = PCA(.95)
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# (Birkaç dk sürebilir) model oluşturuluyor
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=10000)
logisticRegr.fit(train_img, train_lbl)

# Dosya seçme fonksiyonu
def dosya_sec():
    global img_array  # img_array'yi global değişken olarak tanımlayın

    dosya_yolu = filedialog.askopenfilename(initialdir="/", title="Resim dosyasını seç",
                                            filetypes=(("JPEG files", "*.jpg"), ("all files", "*.*")))
    if dosya_yolu:
        img = Image.open(dosya_yolu).convert('L')  # Resmi siyah beyaz formata dönüştürme
        img = img.resize((28, 28))  # Resmi 28x28 piksele boyutlandırma
        imguı = img.resize((100, 100))  # Resmi 28x28 piksele boyutlandırma

        # Scaler'ı sadece training set üzerinde fit yapmamız yeterli..
        img_array = scaler.transform(np.array(img).reshape(1, -1))

        # Ama transform işlemini PCA ile yapıyoruz.
        img_array = pca.transform(img_array)

        # Resmi gösterme
        img_tk = ImageTk.PhotoImage(imguı)
        img_etiketi.configure(image=img_tk)
        img_etiketi.image = img_tk

        # Dosya seçme düğmesi ve tahmin düğmesi konumlarını güncelle
        dosya_sec_dugmesi.grid(row=3, column=0, pady=10, sticky='nsew')
        tahmin_dugmesi.grid(row=3, column=1, pady=10, sticky='nsew')

        # Tahmin ve doğruluk etiketlerini konumlarını güncelle
        tahmin_etiketi.grid(row=1, column=0, columnspan=2, pady=10, sticky='nsew')
        dogruluk_etiketi.grid(row=2, column=0, columnspan=2, pady=10, sticky='nsew')

# Tahmini tetiklemek için düğme
def tahmin_et():
    tahmin = logisticRegr.predict(img_array)
    dogruluk_orani = logisticRegr.score(test_img, test_lbl)

    # Tahmin ve doğruluk etiketlerini güncelleme
    tahmin_etiketi.config(text=f"Tanınan Rakam: {tahmin[0]}", anchor='center')
    dogruluk_etiketi.config(text=f"Modelin Doğruluk Oranı: {dogruluk_orani:.4f}", anchor='center')

# GUI
pencere = tk.Tk()
pencere.title("Rakam Tanıma Arayüzü")

# Arayüzü ortala
pencere_width = 420
pencere_height = 330
screen_width = pencere.winfo_screenwidth()
screen_height = pencere.winfo_screenheight()
x = (screen_width / 2) - (pencere_width / 2)
y = (screen_height / 2) - (pencere_height / 2)
pencere.geometry(f"{pencere_width}x{pencere_height}+{int(x)}+{int(y)}")

# Label widget'ında resmi görüntülemek için bir tkinter PhotoImage oluşturun
img_etiketi = ttk.Label(pencere, anchor='center')
img_etiketi.grid(row=0, column=0, columnspan=2, pady=10, sticky='nsew')

tahmin_etiketi = ttk.Label(pencere, text="Tanınan Rakam: ", font=("Helvetica", 32), anchor='center')
tahmin_etiketi.grid(row=1, column=0, columnspan=2, pady=10, sticky='nsew')

dogruluk_etiketi = ttk.Label(pencere, text="Modelin Doğruluk Oranı: ", font=("Helvetica", 16), anchor='center')
dogruluk_etiketi.grid(row=2, column=0, columnspan=2, pady=10, sticky='nsew')

# Dosya seçme düğmesi
dosya_sec_dugmesi = ttk.Button(pencere, text="Resim Seç", command=dosya_sec)
dosya_sec_dugmesi.grid(row=3, column=0, pady=10, sticky='nsew')

# Tahmini tetiklemek için düğme
tahmin_dugmesi = ttk.Button(pencere, text="Rakamı Tanıma", command=tahmin_et)
tahmin_dugmesi.grid(row=3, column=1, pady=10, sticky='nsew')


# Arayüzü güncelle
pencere.grid_rowconfigure(3, weight=1)
pencere.grid_columnconfigure(0, weight=1)
pencere.grid_columnconfigure(1, weight=1)

pencere.mainloop()
