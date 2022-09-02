import tkinter as tk
from tkinter import filedialog

filetypes = (
    ('Text files', '*.TXT'),
    ('All files', '*.*'),
)

# open-file dialog
root = tk.Tk()
filename = tk.filedialog.askopenfilename(
    title='Select a file...',
    filetypes=filetypes,
)
root.destroy()
print(filename)

# save-as dialog
root = tk.Tk()
filename = tk.filedialog.asksaveasfilename(
    title='Save as...',
    filetypes=filetypes,
    defaultextension='.txt'
)
root.destroy()
kk =tk.filedialog.SaveAs(filename)
print(filename)