import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import pandas as pd
import platform

class CSVEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Editor")
        
        # Create a frame for the navbar
        self.navbar = tk.Frame(root)
        self.navbar.pack(side=tk.TOP, fill=tk.X)
        
        # Create buttons for the navbar
        self.create_button = tk.Button(self.navbar, text="Create CSV", command=self.create_csv)
        self.create_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.load_button = tk.Button(self.navbar, text="Open", command=self.load_csv)
        self.load_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.save_button = tk.Button(self.navbar, text="Save As", command=self.save_csv)
        self.save_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.add_row_button = tk.Button(self.navbar, text="Add Row", command=self.add_row)
        self.add_row_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.delete_row_button = tk.Button(self.navbar, text="Delete Row", command=self.delete_row)
        self.delete_row_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.add_column_button = tk.Button(self.navbar, text="Add Column", command=self.add_column)
        self.add_column_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.remove_column_button = tk.Button(self.navbar, text="Remove Column", command=self.remove_column)
        self.remove_column_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.clear_all_button = tk.Button(self.navbar, text="Clear All", command=self.clear_all)
        self.clear_all_button.pack(side=tk.RIGHT, padx=2, pady=2)
        
        # Create a frame for the treeview
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(self.frame, show="headings", selectmode="extended")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.vsb = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        self.vsb.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=self.vsb.set)
        
        self.hsb = ttk.Scrollbar(self.frame, orient="horizontal", command=self.tree.xview)
        self.hsb.pack(side='bottom', fill='x')
        self.tree.configure(xscrollcommand=self.hsb.set)
        
        # Initialize with a 50x50 grid
        self.df = pd.DataFrame(columns=[f"{i+1}" for i in range(50)],
                               data=[["" for _ in range(50)] for _ in range(50)])
        
        self.history = []  # To store history for undo functionality
        self.display_csv()
        
        # Bind double-click event
        self.tree.bind("<Double-1>", self.on_double_click)
        
        # Bind Ctrl+Z or Cmd+Z for undo functionality
        if platform.system() == 'Darwin':  # macOS
            self.root.bind("<Command-z>", self.undo)
        else:  # Windows and Linux
            self.root.bind("<Control-z>", self.undo)

    def create_csv(self):
        self.save_history()
        self.df = pd.DataFrame({
            "1": [],
            "2": [],
            "3": []
        })
        self.display_csv()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.save_history()
            self.df = pd.read_csv(file_path)
            self.display_csv()

    def save_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.update_df()
            self.df.to_csv(file_path, index=False)
            messagebox.showinfo("Info", "File saved successfully!")

    def display_csv(self):
        self.clear_treeview()
        self.tree["columns"] = ["Row"] + list(self.df.columns)
        self.tree["show"] = "headings"
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        for i, row in enumerate(self.df.itertuples(index=False), start=1):
            self.tree.insert("", "end", values=(i,) + row)
    
    def clear_treeview(self):
        self.tree.delete(*self.tree.get_children())

    def add_row(self):
        self.save_history()
        new_row = ["" for _ in range(len(self.df.columns))]
        self.df.loc[len(self.df)] = new_row
        self.display_csv()

    def delete_row(self):
        selected_item = self.tree.selection()
        if selected_item:
            self.save_history()
            for item in selected_item:
                row_values = self.tree.item(item, "values")[1:]  # exclude row number
                self.df = self.df[self.df.apply(lambda x: list(x) != list(row_values), axis=1)]
            self.display_csv()

    def add_column(self):
        column_name = simpledialog.askstring("Input", "Enter new column name:")
        if column_name:
            self.save_history()
            self.df[column_name] = ""
            self.display_csv()

    def remove_column(self):
        column_name = simpledialog.askstring("Input", "Enter column name to remove:")
        if column_name in self.df.columns:
            self.save_history()
            self.df.drop(columns=[column_name], inplace=True)
            self.display_csv()

    def clear_all(self):
        if messagebox.askyesno("Confirmation", "Are you sure you want to clear all the data?"):
            self.save_history()
            self.df = pd.DataFrame(columns=self.df.columns)
            self.display_csv()
        
    def update_df(self):
        rows = self.tree.get_children()
        data = []
        for row in rows:
            values = self.tree.item(row)["values"][1:]  # exclude row number
            data.append(values)
        self.df = pd.DataFrame(data, columns=self.tree["columns"][1:])  # exclude row number column

    def on_double_click(self, event):
        # Identify the row and column on which the double click occurred
        region = self.tree.identify("region", event.x, event.y)
        if region == "cell":
            column = self.tree.identify_column(event.x)
            row = self.tree.identify_row(event.y)
            
            if column and row:
                column_index = int(column[1:]) - 1  # convert column name to index
                row_index = int(row[1:]) - 1  # convert row name to index

                # Get the current value of the cell
                cell_value = self.tree.set(row, column)
                
                # Create an entry widget at the cell location
                x, y, width, height = self.tree.bbox(row, column)
                self.entry = tk.Entry(self.tree)
                self.entry.place(x=x, y=y, width=width, height=height)
                self.entry.insert(0, cell_value)
                
                # Focus the entry widget and bind the Return key to save the value
                self.entry.focus()
                self.entry.bind("<Return>", lambda event: self.save_edit(row_index, column_index))
                self.entry.bind("<FocusOut>", lambda event: self.save_edit(row_index, column_index))

    def save_edit(self, row_index, column_index):
        # Get the new value from the entry widget
        new_value = self.entry.get()
        
        # Update the DataFrame and the treeview
        column_name = self.tree["columns"][column_index + 1]  # exclude row number column
        self.save_history()
        self.df.at[row_index, column_name] = new_value
        self.tree.set(self.tree.get_children()[row_index], column=column_name, value=new_value)
        
        # Destroy the entry widget
        self.entry.destroy()

    def save_history(self):
        # Save the current state of the DataFrame to history
        self.history.append(self.df.copy())
        if len(self.history) > 10:  # Limit history to last 10 changes
            self.history.pop(0)

    def undo(self, event=None):
        if self.history:
            self.df = self.history.pop()
            self.display_csv()

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVEditor(root)
    root.mainloop()


# import tkinter as tk
# from tkinter import filedialog, messagebox, simpledialog
# from tkinter import ttk
# import pandas as pd
# import platform

# class CSVEditor:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("CSV Editor")
#         self.root.configure(bg="white")
        
#         # Create a frame for the navbar
#         self.navbar = tk.Frame(root, bg="white")
#         self.navbar.pack(side=tk.TOP, fill=tk.X)
        
#         # Create buttons for the navbar
#         self.create_button = tk.Button(self.navbar, text="Create CSV", command=self.create_csv, bg="white", fg="black")
#         self.create_button.pack(side=tk.LEFT, padx=2, pady=2)
        
#         self.load_button = tk.Button(self.navbar, text="Open", command=self.load_csv, bg="white", fg="black")
#         self.load_button.pack(side=tk.LEFT, padx=2, pady=2)
        
#         self.save_button = tk.Button(self.navbar, text="Save As", command=self.save_csv, bg="white", fg="black")
#         self.save_button.pack(side=tk.LEFT, padx=2, pady=2)
        
#         self.add_row_button = tk.Button(self.navbar, text="Add Row", command=self.add_row, bg="white", fg="black")
#         self.add_row_button.pack(side=tk.LEFT, padx=2, pady=2)
        
#         self.delete_row_button = tk.Button(self.navbar, text="Delete Row", command=self.delete_row, bg="white", fg="black")
#         self.delete_row_button.pack(side=tk.LEFT, padx=2, pady=2)
        
#         self.add_column_button = tk.Button(self.navbar, text="Add Column", command=self.add_column, bg="white", fg="black")
#         self.add_column_button.pack(side=tk.LEFT, padx=2, pady=2)
        
#         self.remove_column_button = tk.Button(self.navbar, text="Remove Column", command=self.remove_column, bg="white", fg="black")
#         self.remove_column_button.pack(side=tk.LEFT, padx=2, pady=2)
        
#         self.clear_all_button = tk.Button(self.navbar, text="Clear All", command=self.clear_all, bg="white", fg="black")
#         self.clear_all_button.pack(side=tk.RIGHT, padx=2, pady=2)
        
#         # Create a frame for the treeview
#         self.frame = tk.Frame(root, bg="white")
#         self.frame.pack(fill=tk.BOTH, expand=True)
        
#         self.style = ttk.Style()
#         self.style.configure("Treeview", background="white", foreground="black", fieldbackground="white")
#         self.style.configure("Treeview.Heading", background="white", foreground="black")
#         self.style.map("Treeview.Heading", background=[('active', 'white')], foreground=[('active', 'black')])
#         self.style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])
        
#         self.tree = ttk.Treeview(self.frame, show="headings", selectmode="extended", style="Treeview")
#         self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
#         self.vsb = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
#         self.vsb.pack(side='right', fill='y')
#         self.tree.configure(yscrollcommand=self.vsb.set)
        
#         self.hsb = ttk.Scrollbar(self.frame, orient="horizontal", command=self.tree.xview)
#         self.hsb.pack(side='bottom', fill='x')
#         self.tree.configure(xscrollcommand=self.hsb.set)
        
#         self.tree.tag_configure('grey', background='#f0f0f0')
        
#         # Initialize with a 50x50 grid
#         self.df = pd.DataFrame(columns=[f"{i+1}" for i in range(50)],
#                                data=[["" for _ in range(50)] for _ in range(50)])
        
#         self.history = []  # To store history for undo functionality
#         self.display_csv()
        
#         # Bind double-click event
#         self.tree.bind("<Double-1>", self.on_double_click)
        
#         # Bind Ctrl+Z or Cmd+Z for undo functionality
#         if platform.system() == 'Darwin':  # macOS
#             self.root.bind("<Command-z>", self.undo)
#         else:  # Windows and Linux
#             self.root.bind("<Control-z>", self.undo)

#     def create_csv(self):
#         self.save_history()
#         self.df = pd.DataFrame({
#             "1": [],
#             "2": [],
#             "3": []
#         })
#         self.display_csv()

#     def load_csv(self):
#         file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
#         if file_path:
#             self.save_history()
#             self.df = pd.read_csv(file_path)
#             self.display_csv()

#     def save_csv(self):
#         file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
#         if file_path:
#             self.update_df()
#             self.df.to_csv(file_path, index=False)
#             messagebox.showinfo("Info", "File saved successfully!")

#     def display_csv(self):
#         self.clear_treeview()
#         self.tree["columns"] = ["Row"] + list(self.df.columns)
#         self.tree["show"] = "headings"
#         for col in self.tree["columns"]:
#             self.tree.heading(col, text=col)
#             self.tree.column(col, width=100)
#         for i, row in enumerate(self.df.itertuples(index=False), start=1):
#             tags = ('grey',) if i % 2 == 0 else ()
#             self.tree.insert("", "end", values=(i,) + row, tags=tags)
    
#     def clear_treeview(self):
#         self.tree.delete(*self.tree.get_children())

#     def add_row(self):
#         self.save_history()
#         new_row = ["" for _ in range(len(self.df.columns))]
#         self.df.loc[len(self.df)] = new_row
#         self.display_csv()

#     def delete_row(self):
#         selected_item = self.tree.selection()
#         if selected_item:
#             self.save_history()
#             for item in selected_item:
#                 row_values = self.tree.item(item, "values")[1:]  # exclude row number
#                 self.df = self.df[self.df.apply(lambda x: list(x) != list(row_values), axis=1)]
#             self.display_csv()

#     def add_column(self):
#         column_name = simpledialog.askstring("Input", "Enter new column name:")
#         if column_name:
#             self.save_history()
#             self.df[column_name] = ""
#             self.display_csv()

#     def remove_column(self):
#         column_name = simpledialog.askstring("Input", "Enter column name to remove:")
#         if column_name in self.df.columns:
#             self.save_history()
#             self.df.drop(columns=[column_name], inplace=True)
#             self.display_csv()

#     def clear_all(self):
#         if messagebox.askyesno("Confirmation", "Are you sure you want to clear all the data?"):
#             self.save_history()
#             self.df = pd.DataFrame(columns=self.df.columns)
#             self.display_csv()
        
#     def update_df(self):
#         rows = self.tree.get_children()
#         data = []
#         for row in rows:
#             values = self.tree.item(row)["values"][1:]  # exclude row number
#             data.append(values)
#         self.df = pd.DataFrame(data, columns=self.tree["columns"][1:])  # exclude row number column

#     def on_double_click(self, event):
#         # Identify the row and column on which the double click occurred
# #        region = self.tree.identify("region", event.x, event.y)
#         if region == "cell":
#             column = self.tree.identify_column(event.x)
#             row = self.tree.identify_row(event.y)
            
#             if column and row:
#                 column_index = int(column[1:]) - 1  # convert column name to index
#                 row_index = int(row[1:]) - 1  # convert row name to index

#                 # Get the current value of the cell
#                 cell_value = self.tree.set(row, column)
                
#                 # Create an entry widget at the cell location
#                 x, y, width, height = self.tree.bbox(row, column)
#                 self.entry = tk.Entry(self.tree)
#                 self.entry.place(x=x, y=y, width=width, height=height)
#                 self.entry.insert(0, cell_value)
                
#                 # Focus the entry widget and bind the Return key to save the value
#                 self.entry.focus()
#                 self.entry.bind("<Return>", lambda event: self.save_edit(row_index, column_index))
#                 self.entry.bind("<FocusOut>", lambda event: self.save_edit(row_index, column_index))

#     def save_edit(self, row_index, column_index):
#         # Get the new value from the entry widget
#         new_value = self.entry.get()
        
#         # Update the DataFrame and the treeview
#         column_name = self.tree["columns"][column_index + 1]  # exclude row number column
#         self.save_history()
#         self.df.at[row_index, column_name] = new_value
#         self.tree.set(self.tree.get_children()[row_index], column=column_name, value=new_value)
#         # Destroy the entry widget
#         self.entry.destroy()

#     def save_history(self):
#         # Save the current state of the DataFrame to history
#         self.history.append(self.df.copy())
#         if len(self.history) > 10:  # Limit history to last 10 changes
#             self.history.pop(0)

#     def undo(self, event=None):
#         if self.history:
#             self.df = self.history.pop()
#             self.display_csv()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = CSVEditor(root)
#     root.mainloop()
