import tkinter as tk 
from tkinter import filedialog 
print("before dialog") 
root = tk.Tk() 
root.withdraw() 
root.attributes("-topmost", True) 
folder = filedialog.askdirectory(title="Pick a folder") 
print("You picked:", folder) 
