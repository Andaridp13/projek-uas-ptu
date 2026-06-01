import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
import librosa
import pickle
import os
import pyaudio
import wave
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame
import tempfile
import uuid
import subprocess

# setting tema tampilan light mode
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue") 

# konfigurasi path dan parameter audio buat rekam mic
FOLDER_MODEL = r'D:\semester 6\projek_ptu'
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024
DURASI_REKAM = 2.0
FILE_SEMENTARA = 'rekaman_sementara.wav'
ONSET_PRE = 0.1
DURASI_TRIM = 1.0
CONFIDENCE_THRESHOLD = 50.0

# hide terminal bawaan edge-tts biar ui tetep clean pas render (windows only)
HIDE_CMD = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

# load model svm dan scaler yang udah ditraining sebelumnya
with open(os.path.join(FOLDER_MODEL, 'model_svm.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(FOLDER_MODEL, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)
with open(os.path.join(FOLDER_MODEL, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# inisialisasi mixer buat muter suara hasil tts
pygame.mixer.init()

def get_file_temp(ekstensi='mp3'):
    # generate nama file random di folder temp biar ga nabrak file lama
    return os.path.join(tempfile.gettempdir(), f'tts_{uuid.uuid4().hex}.{ekstensi}')

def ekstrak_fitur(audio, sr):
    # normalisasi volume suara inputan
    audio = librosa.util.normalize(audio)
    
    # cari titik peak suara pertama biar ga kepotong noise di awal
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=512, backtrack=True, units='samples')
    if len(onset_frames) > 0:
        onset_sample = onset_frames[0]
        start = max(0, onset_sample - int(ONSET_PRE * sr))
        end = min(len(audio), start + int(DURASI_TRIM * sr))
        audio_trim = audio[start:end]
        # balikin ke audio awal kalau hasil potongannya kekecilan
        if len(audio_trim) < int(0.3 * sr):
            audio_trim = audio
    else:
        audio_trim = audio
        
    # hitung array fitur mfcc dll buat masukin ke model svm
    mfcc = librosa.feature.mfcc(y=audio_trim, sr=sr, n_mfcc=40)
    mfcc_rata = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_rata = np.mean(delta_mfcc, axis=1)
    
    chroma = librosa.feature.chroma_stft(y=audio_trim, sr=sr)
    chroma_rata = np.mean(chroma, axis=1)
    
    zcr = librosa.feature.zero_crossing_rate(y=audio_trim)
    zcr_rata = np.mean(zcr)
    
    # gabungin semua matriks fiturnya jadi satu array 1d
    fitur = np.concatenate([mfcc_rata, mfcc_std, delta_rata, chroma_rata, [zcr_rata]])
    return fitur, mfcc

def putar_audio(path_file):
    # unload file yang lama dulu sebelum play yang baru
    pygame.mixer.music.unload()
    pygame.mixer.music.load(path_file)
    pygame.mixer.music.play()
    # block thread ini sampai lagu selesai diputar
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def update_gui_aman(hasil, confidence, mfcc, perlu_ulang):
    # update text hasil prediksi warna
    if perlu_ulang:
        label_hasil.configure(text='ULANGI', text_color='#E74C3C')
        label_confidence.configure(text=f'Suara kurang jelas ({confidence:.2f}%)', text_color='#E74C3C')
    else:
        label_hasil.configure(text=f'{hasil.upper()}', text_color='#27AE60')
        label_confidence.configure(text=f'Confidence: {confidence:.2f}%', text_color='#7F8C8D')
        app.hasil_asr = hasil
        tombol_kirim_tts.configure(state='normal')
        
    # bersihin canvas plot lama
    for widget in frame_plot.winfo_children():
        widget.destroy()
        
    # render grafik spektogram mfcc buat di display, ukuran digedein buat proyektor
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor='white')
    ax.set_facecolor('white')
    ax.tick_params(colors='#2C3E50', labelsize=10)
    
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax, cmap='viridis')
    cbar = plt.colorbar(img, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='#2C3E50')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#2C3E50')
    
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    plt.close(fig)
    
    tombol_rekam.configure(state='normal', text='🔴 Rekam Suara', fg_color='#E84393')

def rekam_dan_prediksi():
    # disable tombol biar ga error keklik dua kali pas lg jalan
    tombol_rekam.configure(state='disabled', text='Mendengarkan...', fg_color='#95A5A6')
    tombol_kirim_tts.configure(state='disabled')
    label_hasil.configure(text='Memproses...', text_color='#2980B9')
    label_confidence.configure(text='')
    
    def proses():
        # mulai record dari mic laptop
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []
        for _ in range(0, int(RATE / CHUNK * DURASI_REKAM)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # save raw datanya ke file temporary
        with wave.open(FILE_SEMENTARA, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            
        # load file yg barusan direkam terus di ekstrak
        audio, sr = librosa.load(FILE_SEMENTARA, sr=RATE)
        fitur, mfcc = ekstrak_fitur(audio, sr)
        fitur_scaled = scaler.transform(fitur.reshape(1, -1))
        
        # ambil hasil prediksi sama tingkat konfidennya
        probabilitas = model.predict_proba(fitur_scaled)[0]
        indeks_terbaik = np.argmax(probabilitas)
        hasil = le.classes_[indeks_terbaik]
        confidence = probabilitas[indeks_terbaik] * 100
        perlu_ulang = confidence < CONFIDENCE_THRESHOLD
        
        # lempar balik ke main thread gui
        app.after(0, update_gui_aman, hasil, confidence, mfcc, perlu_ulang)
        
    # proses rekamnya dilempar ke background thread
    threading.Thread(target=proses, daemon=True).start()

def kirim_ke_tts():
    # pass string hasil asr ke kolom text tab tts
    if hasattr(app, 'hasil_asr'):
        entry_tts.delete("0.0", tk.END)
        entry_tts.insert("0.0", app.hasil_asr)
        tabview.set('🔊 Teks ke Suara (TTS)')
        jalankan_tts()

def jalankan_tts():
    # proses konversi text jadi voice via edge tts
    teks = entry_tts.get("0.0", tk.END).strip()
    if not teks:
        messagebox.showwarning("Peringatan", "Teks tidak boleh kosong")
        return
    kecepatan = var_kecepatan.get()
    gender = var_gender.get()
    
    # sesuaikan model suara sama input user
    voice = 'id-ID-ArdiNeural' if gender == 'laki' else 'id-ID-GadisNeural'
    rate = '+0%' if kecepatan == 'normal' else ('-25%' if kecepatan == 'slow' else '+25%')
    
    tombol_putar.configure(state='disabled', text='Memproses...', fg_color='#95A5A6')
    
    def proses_tts():
        try:
            pygame.mixer.music.unload()
            file_mp3 = get_file_temp('mp3')
            # call api local cli nya pake subprocess
            cmd = ['edge-tts', '--voice', voice, f'--rate={rate}', '--text', teks, '--write-media', file_mp3]
            subprocess.run(cmd, check=True, creationflags=HIDE_CMD)
            putar_audio(file_mp3)
            app.after(0, lambda: tombol_putar.configure(state='normal', text='▶ Putar Suara', fg_color='#3498DB'))
        except Exception as e:
            app.after(0, lambda: messagebox.showerror("Error", f"Gagal memproses TTS: {e}"))
            app.after(0, lambda: tombol_putar.configure(state='normal', text='▶ Putar Suara', fg_color='#3498DB'))
            
    threading.Thread(target=proses_tts, daemon=True).start()

def simpan_audio():
    # logic buat nyimpen file mp3 hasil tts ke komputer
    teks = entry_tts.get("0.0", tk.END).strip()
    if not teks:
        messagebox.showwarning("Peringatan", "Teks tidak boleh kosong")
        return
    kecepatan = var_kecepatan.get()
    gender = var_gender.get()
    
    path_simpan = filedialog.asksaveasfilename(
        defaultextension=".mp3",
        filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")],
        title="Simpan Audio"
    )
    if not path_simpan:
        return
        
    voice = 'id-ID-ArdiNeural' if gender == 'laki' else 'id-ID-GadisNeural'
    rate = '+0%' if kecepatan == 'normal' else ('-25%' if kecepatan == 'slow' else '+25%')
    
    try:
        cmd = ['edge-tts', '--voice', voice, f'--rate={rate}', '--text', teks, '--write-media', path_simpan]
        subprocess.run(cmd, check=True, creationflags=HIDE_CMD)
        messagebox.showinfo("Berhasil", f"Audio berhasil disimpan di:\n{path_simpan}")
    except Exception as e:
        messagebox.showerror("Error", f"Gagal menyimpan audio: {e}")

# init aplikasi ctk, disetting lebar dikit buat nyesuaiin resolusi proyektor
app = ctk.CTk()
app.title('ASR & TTS Studio')
app.geometry('950x750')
app.configure(fg_color="#F4F6F7")
app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(1, weight=1)

# frame header atas 
header_frame = ctk.CTkFrame(app, fg_color="transparent")
header_frame.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="ew")
ctk.CTkLabel(header_frame, text='🎙️ ASR & TTS Studio', font=ctk.CTkFont(family='Segoe UI', size=32, weight='bold'), text_color="#2C3E50").pack(side="left")
ctk.CTkLabel(header_frame, text='Sistem Cerdas Pengenalan Suara', font=ctk.CTkFont(family='Segoe UI', size=18), text_color="#7F8C8D").pack(side="left", padx=15, pady=(10,0))

# tab layout menu
tabview = ctk.CTkTabview(app, width=850, height=600, corner_radius=15, fg_color="white", 
                         segmented_button_selected_color="#E84393", 
                         segmented_button_selected_hover_color="#FD79A8",
                         text_color="#2C3E50")
tabview.grid(row=1, column=0, padx=20, pady=15, sticky="nsew")

# cara bener buat ngegedein font tab di customtkinter versi baru
tabview._segmented_button.configure(font=ctk.CTkFont(size=16, weight="bold"))

tab_asr = tabview.add('💬 Suara ke Teks (ASR)')
tab_tts = tabview.add('🔊 Teks ke Suara (TTS)')

# setup page asr
tab_asr.grid_columnconfigure(0, weight=1)
tab_asr.grid_columnconfigure(1, weight=1)
tab_asr.configure(fg_color="white")

# panel tombol rekam
frame_kontrol = ctk.CTkFrame(tab_asr, corner_radius=15, fg_color="white", border_width=1, border_color="#E2E8F0")
frame_kontrol.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
ctk.CTkLabel(frame_kontrol, text='⚙️ Kontrol Rekaman', font=ctk.CTkFont(size=20, weight='bold'), text_color="#2C3E50").pack(pady=(30, 15))
tombol_rekam = ctk.CTkButton(frame_kontrol, text='🔴 Rekam Suara', font=ctk.CTkFont(size=18, weight='bold'), fg_color='#E84393', hover_color='#FD79A8', height=55, command=rekam_dan_prediksi)
tombol_rekam.pack(pady=20, padx=30, fill='x')

# panel nampilin teks hasil deteksi
frame_hasil = ctk.CTkFrame(tab_asr, corner_radius=15, fg_color="white", border_width=1, border_color="#E2E8F0")
frame_hasil.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
ctk.CTkLabel(frame_hasil, text='🎯 Hasil Prediksi', font=ctk.CTkFont(size=20, weight='bold'), text_color="#2C3E50").pack(pady=(20, 10))
# ukuran font hasil sengaja digedein max biar nampak pas presentasi
label_hasil = ctk.CTkLabel(frame_hasil, text='Menunggu...', font=ctk.CTkFont(size=48, weight='bold'), text_color='#7F8C8D')
label_hasil.pack(pady=10)
label_confidence = ctk.CTkLabel(frame_hasil, text='-', font=ctk.CTkFont(size=16), text_color='#95A5A6')
label_confidence.pack()
tombol_kirim_tts = ctk.CTkButton(frame_hasil, text='Kirim ke TTS ➔', font=ctk.CTkFont(size=16, weight='bold'), fg_color="#F39C12", hover_color="#E67E22", height=40, command=kirim_ke_tts, state='disabled')
tombol_kirim_tts.pack(pady=20)

# panel buat nampilin grafik spektrum suaranya
frame_grafik = ctk.CTkFrame(tab_asr, corner_radius=15, fg_color="white", border_width=1, border_color="#E2E8F0")
frame_grafik.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
ctk.CTkLabel(frame_grafik, text='📊 Visualisasi Spektogram (MFCC)', font=ctk.CTkFont(size=16, weight='bold'), text_color="#2C3E50").pack(pady=(10, 0))
frame_plot = ctk.CTkFrame(frame_grafik, fg_color="transparent")
frame_plot.pack(fill="both", expand=True, padx=10, pady=10)

# setup page tts
tab_tts.grid_columnconfigure(0, weight=1)
tab_tts.configure(fg_color="white")

ctk.CTkLabel(tab_tts, text='📝 Teks Input', font=ctk.CTkFont(size=20, weight='bold'), text_color="#2C3E50").pack(pady=(15, 5), anchor="w", padx=20)
entry_tts = ctk.CTkTextbox(tab_tts, height=130, font=ctk.CTkFont(size=18), corner_radius=10, border_width=2, border_color="#E2E8F0", fg_color="#F8F9FA", text_color="#2C3E50")
entry_tts.pack(padx=20, fill='x', pady=(0, 15))

# panel konfigurasi suara tts
frame_settings = ctk.CTkFrame(tab_tts, corner_radius=15, fg_color="white", border_width=1, border_color="#E2E8F0")
frame_settings.pack(padx=20, fill='x', pady=5)
frame_settings.grid_columnconfigure(1, weight=1)

ctk.CTkLabel(frame_settings, text='⚙️ Pengaturan Suara', font=ctk.CTkFont(size=18, weight='bold'), text_color="#2C3E50").grid(row=0, column=0, columnspan=2, pady=(15, 10), padx=20, sticky="w")

ctk.CTkLabel(frame_settings, text='Kecepatan:', font=ctk.CTkFont(size=16), text_color="#2C3E50").grid(row=1, column=0, padx=20, pady=10, sticky="w")
var_kecepatan = ctk.StringVar(value='normal')
frame_speed = ctk.CTkFrame(frame_settings, fg_color="transparent")
frame_speed.grid(row=1, column=1, sticky="w")
ctk.CTkRadioButton(frame_speed, text='Lambat', variable=var_kecepatan, value='slow', text_color="#2C3E50", font=ctk.CTkFont(size=16)).pack(side="left", padx=(0,15))
ctk.CTkRadioButton(frame_speed, text='Normal', variable=var_kecepatan, value='normal', text_color="#2C3E50", font=ctk.CTkFont(size=16)).pack(side="left", padx=15)
ctk.CTkRadioButton(frame_speed, text='Cepat', variable=var_kecepatan, value='fast', text_color="#2C3E50", font=ctk.CTkFont(size=16)).pack(side="left", padx=15)

ctk.CTkLabel(frame_settings, text='Gender:', font=ctk.CTkFont(size=16), text_color="#2C3E50").grid(row=2, column=0, padx=20, pady=(10, 20), sticky="w")
var_gender = ctk.StringVar(value='perempuan')
frame_gender = ctk.CTkFrame(frame_settings, fg_color="transparent")
frame_gender.grid(row=2, column=1, sticky="w", pady=(0,10))
ctk.CTkRadioButton(frame_gender, text='Perempuan (Gadis)', variable=var_gender, value='perempuan', text_color="#2C3E50", font=ctk.CTkFont(size=16)).pack(side="left", padx=(0,15))
ctk.CTkRadioButton(frame_gender, text='Laki-laki (Ardi)', variable=var_gender, value='laki', text_color="#2C3E50", font=ctk.CTkFont(size=16)).pack(side="left", padx=15)

# panel tombol buat play sama save
frame_actions = ctk.CTkFrame(tab_tts, fg_color="transparent")
frame_actions.pack(pady=25)
tombol_putar = ctk.CTkButton(frame_actions, text='▶ Putar Suara', font=ctk.CTkFont(size=16, weight='bold'), fg_color="#3498DB", hover_color="#2980B9", height=50, width=160, command=jalankan_tts)
tombol_putar.pack(side='left', padx=10)
tombol_simpan = ctk.CTkButton(frame_actions, text='💾 Simpan Audio', font=ctk.CTkFont(size=16, weight='bold'), fg_color="#2C3E50", hover_color="#34495E", height=50, width=160, command=simpan_audio)
tombol_simpan.pack(side='left', padx=10)

# ngeloop main thread gui biar ga force close
app.mainloop()