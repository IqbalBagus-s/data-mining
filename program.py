import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Memuat data dari file Excel
data_set = pd.read_excel("data-monitor.xlsx")
# Menampilkan data teratas
print(data_set.head(10))

# Informasi data
print(data_set.info())

# Memeriksa apakah dataset kosong
print("Apakah data Kosong : ", data_set.empty)

# Inisialisasi LabelEncoder untuk setiap kolom
en_screen = LabelEncoder()
en_budget = LabelEncoder()
en_title = LabelEncoder()
en_brand = LabelEncoder()
en_resolution = LabelEncoder()
en_aspect_ratio = LabelEncoder()

# Fungsi untuk memastikan semua nilai dalam kolom adalah string
def convert_to_string(series):
    return series.astype(str)

# Konversi kolom yang diperlukan ke string
data_set['Screen Size'] = convert_to_string(data_set['Screen Size'])
data_set['Kelas Budget'] = convert_to_string(data_set['Kelas Budget'])
data_set['Title'] = convert_to_string(data_set['Title'])
data_set['Brand'] = convert_to_string(data_set['Brand'])
data_set['Resolution'] = convert_to_string(data_set['Resolution'])
data_set['Aspect Ratio'] = convert_to_string(data_set['Aspect Ratio'])

# Transformasi data pada kolom
data_set['Screen Size'] = en_screen.fit_transform(data_set['Screen Size'])
data_set['Kelas Budget'] = en_budget.fit_transform(data_set['Kelas Budget'])
data_set['Title'] = en_title.fit_transform(data_set['Title'])
data_set['Brand'] = en_brand.fit_transform(data_set['Brand'])
data_set['Resolution'] = en_resolution.fit_transform(data_set['Resolution'])
data_set['Aspect Ratio'] = en_aspect_ratio.fit_transform(data_set['Aspect Ratio'])

print(data_set.head(10))

# Memisahkan fitur dan label
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

# Membagi dataset menjadi data pelatihan dan data pengujian
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

print(f"x_train = {len(x_train)}")
print(f"x_test = {len(x_test)}")
print(f"y_train = {len(y_train)}")
print(f"y_test = {len(y_test)}")

# Membuat scaler untuk fitur numerik
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Membuat dan melatih model Naive Bayes
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Melakukan prediksi pada data uji
y_pred = classifier.predict(x_test)

# Menghitung dan menampilkan matriks kebingungan
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Evaluasi model dengan classification report
akurasi = classification_report(y_test, y_pred)
print(akurasi)

# Menghitung dan menampilkan skor akurasi
skor_akurasi = accuracy_score(y_test, y_pred)
print("tingkat akurasi = %d persen" % (skor_akurasi * 100))

# Menyimpan hasil prediksi ke dalam DataFrame
ydata = pd.DataFrame()
ydata['y_test'] = pd.DataFrame(y_test)
ydata['y_pred'] = pd.DataFrame(y_pred)

print(ydata)
# Menyimpan DataFrame ke file Excel
# ydata.to_excel('data_prediksi.xlsx', index=False)
