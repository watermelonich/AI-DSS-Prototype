import re
import tkinter as tk 
from tkinter.scrolledtext import ScrolledText
import nltk
from nltk.corpus import words

nltk.download ("words")
class SpellingChecker:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("600x500")
        self.text = ScrolledText(self.root, font=("Arial", 14))
        self.text