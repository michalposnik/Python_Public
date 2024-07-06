import os
import random
import numpy as np
from PIL import Image, ImageTk
import joblib
import tkinter as tk
from tkinter import messagebox

def load_and_prepare_image(image_path):
    img = Image.open(image_path).convert('L')  # Konwersja do skali szarości
    img = img.resize((64, 64))  # Zmiana rozmiaru do 64x64
    img_array = np.array(img).flatten()  # Spłaszczenie obrazu
    return img_array

def user_feedback(image_path, ai_label, feedback):
    log_file_path = 'C:/"TWOJA_ŚCIEżKA"/model_AI_do_kategoryzowania_obrazow_GitHub/feedback_log.txt'  # Podaj ścieżkę do folderu zawsze / prawy ukośnik to ścieżka do pliku logu

    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as file:
            pass  # Utworzenie pliku, jeśli nie istnieje

    # Odczytanie istniejących logów
    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()

    # Sprawdzenie, czy istnieje wpis dla tego obrazu
    entry_exists = False
    entry_index = -1
    for i, line in enumerate(log_lines):
        if image_path in line:
            entry_exists = True
            entry_index = i
            break

    if feedback == "Prawda":
        new_entry = f"Poprawna klasyfikacja dla {image_path}: {ai_label}\n"
        print(f"Użytkownik potwierdził poprawność klasyfikacji AI dla {image_path} jako poprawną ({ai_label}).")
    else:
        correct_label = "NIE" if ai_label == "TAK" else "TAK"
        new_entry = f"Niepoprawna klasyfikacja dla {image_path}: {correct_label}\n"
        print(f"Użytkownik oznaczył klasyfikację AI dla {image_path} jako niepoprawną. Poprawna etykieta to {correct_label}.")

    # Jeśli wpis istnieje, zastąp go nowym wpisem
    if entry_exists:
        log_lines[entry_index] = new_entry
    else:
        log_lines.append(new_entry)

    # Zapisanie zmodyfikowanej listy logów do pliku
    with open(log_file_path, 'w') as file:
        file.writelines(log_lines)
def classify_images(model, scaler, image_paths):
    classifications = []
    for image_path in image_paths:
        img_array = load_and_prepare_image(image_path)
        img_array_scaled = scaler.transform([img_array])
        prediction = model.predict(img_array_scaled)
        label = "TAK" if prediction == 1 else "NIE"
        classifications.append((image_path, label))
    return classifications

def display_images_with_classifications(classifications, on_done):
    root = tk.Tk()
    root.title("Wyniki klasyfikacji obrazów")

    def next_set():
        root.destroy()
        on_done()

    next_set_button = tk.Button(root, text="Następny zestaw", command=next_set)
    next_set_button.pack(side=tk.TOP, pady=10)

    for i, (image_path, label) in enumerate(classifications):
        frame = tk.Frame(root)
        frame.pack(side=tk.TOP, pady=10)

        img = Image.open(image_path)
        img = img.resize((200, 200))  # Zmiana rozmiaru do wyświetlenia
        img_tk = ImageTk.PhotoImage(img)

        img_label = tk.Label(frame, image=img_tk)
        img_label.image = img_tk  # Zachowanie odniesienia do obrazka
        img_label.pack(side=tk.LEFT)

        label_label = tk.Label(frame, text=label, font=("Helvetica", 16))
        label_label.pack(side=tk.LEFT, padx=10)

        correct_button = tk.Button(frame, text="Zgodność",
                                   command=lambda img=image_path, ai_lbl=label: user_feedback(img, ai_lbl, "Prawda"))
        correct_button.pack(side=tk.LEFT, padx=5)

        incorrect_button = tk.Button(frame, text="Zmień",
                                     command=lambda img=image_path, ai_lbl=label: user_feedback(img, ai_lbl, "Fałsz"))
        incorrect_button.pack(side=tk.LEFT, padx=5)

    root.mainloop()

def main():
    model_path = 'svm_model.pkl'
    scaler_path = 'scaler.pkl'

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    test_folder = 'C:/"TWOJA_ŚCIEżKA"/model_AI_do_kategoryzowania_obrazow_GitHub/zdjęcia python/testowy_wszystkie'  # Podaj ścieżkę do folderu zawsze / prawy ukośnik
    image_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if
                   f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

    if len(image_files) < 5:
        print("Za mało obrazów w folderze testowym. Potrzebne są przynajmniej 4 obrazów.")
        return

    def show_next_set():
        if len(image_files) < 5:
            messagebox.showinfo("Koniec obrazów",
                                "Za mało obrazów w folderze testowym do wyświetlenia kolejnego zestawu.")
            return
        random_images = random.sample(image_files, 4)
        classifications = classify_images(model, scaler, random_images)
        display_images_with_classifications(classifications, show_next_set)

    show_next_set()

if __name__ == "__main__":
    main()


