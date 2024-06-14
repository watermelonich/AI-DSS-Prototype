# import tkinter as tk
# from tkinter import ttk
# from tkinter import font as tkFont
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from gui_mainboard_pages import DashboardPage, ReportsPage, DataAnalysisPage, ModelSelectionPage, DataManagementPage

# class AIDSSApp(tk.Tk):
#     def __init__(self):
#         super().__init__()

#         self.title("AI-DSS")
#         self.geometry("1200x800")
#         self.configure(bg='white')

#         # Top-level container frame
#         container = tk.Frame(self, bg='white')
#         container.pack(expand=True)

#         # Top bar frame
#         top_bar = tk.Frame(container, bg='white', height=60)
#         top_bar.pack(side='top', fill='x')

#         profile_icon = tk.Label(top_bar, text='Profile', bg='white', fg='white')
#         profile_icon.pack(side='right', padx=20)

#         chatbot_icon = tk.Label(top_bar, text='Chatbot', bg='white', fg='white')
#         chatbot_icon.pack(side='right', padx=20)

#         menu_icon = tk.Label(top_bar, text='Menu', bg='white', fg='white')
#         menu_icon.pack(side='right', padx=20)

#         # Center container for sidebar and mainboard
#         center_container = tk.Frame(container, bg='white')
#         center_container.pack(expand=True, fill='both')

#         # Sidebar frame
#         sidebar = tk.Frame(center_container, bg='#1c4a8b', width=200, height=600)
#         sidebar.pack(side='left', fill='y', padx=10, pady=10)
#         sidebar.pack_propagate(0)  # Prevents frame from shrinking to content size

#         self.pages = {
#             "Dashboard": DashboardPage,
#             "Reports": ReportsPage,
#             "Data Analysis": DataAnalysisPage,
#             "Model Selection": ModelSelectionPage,
#             "Data Management": DataManagementPage
#         }

#         buttons = ["Dashboard", "Reports", "Data Analysis", "Model Selection", "Data Management"]

#         for btn_text in buttons:
#             btn = tk.Button(sidebar, text=btn_text, bg='#1c4a8b', fg='black', relief='flat',
#                             command=lambda name=btn_text: self.show_page(name))
#             btn.pack(fill='x', pady=10)

#         # Create a canvas and scrollbar for the main board
#         canvas = tk.Canvas(center_container, bg='white', width=800, height=600)
#         canvas.pack(side='right', expand=True, fill='both', padx=10, pady=10)

#         scrollbar = ttk.Scrollbar(center_container, orient='vertical', command=canvas.yview)
#         scrollbar.pack(side='right', fill='y')

#         canvas.configure(yscrollcommand=scrollbar.set)
#         canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

#         self.main_board = tk.Frame(canvas, bg='white', width=800, height=600)
#         self.main_board.pack(expand=True, fill='both')
#         self.main_board.pack_propagate(0)  # Prevents frame from shrinking to content size

#         canvas.create_window((0, 0), window=self.main_board, anchor='nw')

#         self.frames = {}
#         for name, Page in self.pages.items():
#             frame = Page(parent=self.main_board, controller=self)
#             self.frames[name] = frame
#             frame.grid(row=0, column=0, sticky="nsew")

#         self.show_page("Dashboard")

#     def show_page(self, page_name):
#         frame = self.frames[page_name]
#         frame.tkraise()

#     def create_plot(self, parent, title, side):
#         # Create a matplotlib figure
#         fig = Figure(figsize=(5, 4), dpi=100)
#         ax = fig.add_subplot(111)
#         ax.set_title(title)
#         ax.plot([0, 1, 2, 3], [0, 1, 4, 9])

#         # Create a Tkinter canvas to host the plot
#         canvas = FigureCanvasTkAgg(fig, master=parent)
#         canvas.draw()
#         canvas.get_tk_widget().pack(side=side, padx=10)

#         return canvas

# if __name__ == "__main__":
#     app = AIDSSApp()
#     app.mainloop()






# import tkinter as tk
# from tkinter import ttk
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from gui_mainboard_pages import DashboardPage, ReportsPage, DataAnalysisPage, ModelSelectionPage, DataManagementPage
# from PIL import Image, ImageTk

# class AIDSSApp(tk.Tk):
#     def __init__(self):
#         super().__init__()

#         self.title("AI-DSS")
#         self.geometry("1200x800")
#         self.configure(bg='white')

#         bg_image_path = "../assets/canvasbg.png"
#         bg_image = Image.open(bg_image_path)
#         bg_render = ImageTk.PhotoImage(bg_image)

#         # Top-level container frame
#         container = tk.Frame(self, bg='white')
#         container.pack(expand=True)

#         # Top bar frame
#         top_bar = tk.Frame(container, bg='white', height=60)
#         top_bar.pack(side='top', fill='x')

#         profile_icon = tk.Label(top_bar, text='Profile', bg='white', fg='white')
#         profile_icon.pack(side='right', padx=20)

#         chatbot_icon = tk.Label(top_bar, text='Chatbot', bg='white', fg='white')
#         chatbot_icon.pack(side='right', padx=20)

#         menu_icon = tk.Label(top_bar, text='Menu', bg='white', fg='white')
#         menu_icon.pack(side='right', padx=20)

#         # Center container for sidebar and mainboard
#         center_container = tk.Frame(container, bg='white')
#         center_container.pack(expand=True, fill='both')

#         # Sidebar frame
#         sidebar = tk.Frame(center_container, bg='white', width=200, height=600)
#         sidebar.pack(side='left', fill='y', padx=10, pady=10)
#         sidebar.pack_propagate(0)  # Prevents frame from shrinking to content size

#         self.pages = {
#             "Dashboard": DashboardPage,
#             "Reports": ReportsPage,
#             "Data Analysis": DataAnalysisPage,
#             "Model Selection": ModelSelectionPage,
#             "Data Management": DataManagementPage
#         }

#         buttons = ["Dashboard", "Reports", "Data Analysis", "Model Selection", "Data Management"]

#         for btn_text in buttons:
#             btn = tk.Button(sidebar, text=btn_text, bg='#1c4a8b', fg='black', relief='flat',
#                             command=lambda name=btn_text: self.show_page(name))
#             btn.pack(fill='x', pady=10)

#         # Create a canvas and scrollbar for the main board
#         canvas = tk.Canvas(center_container, bg='white', width=800, height=600)
#         canvas.pack(side='right', expand=True, fill='both', padx=10, pady=10)

#         scrollbar = ttk.Scrollbar(center_container, orient='vertical', command=canvas.yview)
#         scrollbar.pack(side='right', fill='y')

#         canvas.configure(yscrollcommand=scrollbar.set)
#         canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

#         self.main_board = tk.Frame(canvas, bg='white', width=800, height=600)
#         self.main_board.pack(expand=True, fill='both')
#         self.main_board.pack_propagate(0)  # Prevents frame from shrinking to content size

#         canvas.create_window((0, 0), window=self.main_board, anchor='nw')

#         self.frames = {}
#         for name, Page in self.pages.items():
#             frame = Page(parent=self.main_board, controller=self)
#             self.frames[name] = frame
#             frame.grid(row=0, column=0, sticky="nsew")

#         self.show_page("Dashboard")

#     def show_page(self, page_name):
#         frame = self.frames[page_name]
#         frame.tkraise()

#     def create_plot(self, parent, title, side):
#         # Create a matplotlib figure
#         fig = Figure(figsize=(5, 4), dpi=100)
#         ax = fig.add_subplot(111)
#         ax.set_title(title)
#         ax.plot([0, 1, 2, 3], [0, 1, 4, 9])

#         # Create a Tkinter canvas to host the plot
#         canvas = FigureCanvasTkAgg(fig, master=parent)
#         canvas.draw()
#         canvas.get_tk_widget().pack(side=side, padx=10)

#         return canvas

# if __name__ == "__main__":
#     app = AIDSSApp()
#     app.mainloop()









# import tkinter as tk
# from tkinter import ttk
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from gui_mainboard_pages import DashboardPage, ReportsPage, DataAnalysisPage, ModelSelectionPage, DataManagementPage
# from PIL import Image, ImageTk

# class AIDSSApp(tk.Tk):
#     def __init__(self):
#         super().__init__()

#         self.title("AI-DSS")
#         self.geometry("1200x800")
#         self.configure(bg='white')

#         bg_image_path = "../assets/canvasbg.png"
#         bg_image = Image.open(bg_image_path)
#         self.bg_render = ImageTk.PhotoImage(bg_image)  # Store the reference to avoid garbage collection

#         # Set the background label
#         background_label = tk.Label(self, image=self.bg_render)
#         background_label.place(relwidth=1, relheight=1)

#         # Top-level container frame
#         container = tk.Frame(self, bg='white')
#         container.pack(expand=True)

#         # Top bar frame
#         top_bar = tk.Frame(container, bg='white', height=60)
#         top_bar.pack(side='top', fill='x')

#         profile_icon = tk.Label(top_bar, text='Profile', fg='white')
#         profile_icon.pack(side='right', padx=20)

#         chatbot_icon = tk.Label(top_bar, text='Chatbot', fg='white')
#         chatbot_icon.pack(side='right', padx=20)

#         menu_icon = tk.Label(top_bar, text='Menu', fg='white')
#         menu_icon.pack(side='right', padx=20)

#         # Center container for sidebar and mainboard
#         center_container = tk.Frame(container)
#         center_container.pack(expand=True, fill='both')

#         # Sidebar frame
#         sidebar = tk.Frame(center_container, bg='white', width=200, height=600)
#         sidebar.pack(side='left', fill='y', padx=10, pady=10)
#         sidebar.pack_propagate(0)  # Prevents frame from shrinking to content size

#         self.pages = {
#             "Dashboard": DashboardPage,
#             "Reports": ReportsPage,
#             "Data Analysis": DataAnalysisPage,
#             "Model Selection": ModelSelectionPage,
#             "Data Management": DataManagementPage
#         }

#         buttons = ["Dashboard", "Reports", "Data Analysis", "Model Selection", "Data Management"]

#         for btn_text in buttons:
#             btn = tk.Button(sidebar, text=btn_text, bg='#1c4a8b', fg='black', relief='flat',
#                             command=lambda name=btn_text: self.show_page(name))
#             btn.pack(fill='x', pady=10)

#         # Create a canvas and scrollbar for the main board
#         canvas = tk.Canvas(center_container, bg='white', width=800, height=600)
#         canvas.pack(side='right', expand=True, fill='both', padx=10, pady=10)

#         scrollbar = ttk.Scrollbar(center_container, orient='vertical', command=canvas.yview)
#         scrollbar.pack(side='right', fill='y')

#         canvas.configure(yscrollcommand=scrollbar.set)
#         canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

#         self.main_board = tk.Frame(canvas, bg='white', width=800, height=600)
#         self.main_board.pack(expand=True, fill='both')
#         self.main_board.pack_propagate(0)  # Prevents frame from shrinking to content size

#         canvas.create_window((0, 0), window=self.main_board, anchor='nw')

#         self.frames = {}
#         for name, Page in self.pages.items():
#             frame = Page(parent=self.main_board, controller=self)
#             self.frames[name] = frame
#             frame.grid(row=0, column=0, sticky="nsew")

#         self.show_page("Dashboard")

#     def show_page(self, page_name):
#         frame = self.frames[page_name]
#         frame.tkraise()

#     def create_plot(self, parent, title, side):
#         # Create a matplotlib figure
#         fig = Figure(figsize=(5, 4), dpi=100)
#         ax = fig.add_subplot(111)
#         ax.set_title(title)
#         ax.plot([0, 1, 2, 3], [0, 1, 4, 9])

#         # Create a Tkinter canvas to host the plot
#         canvas = FigureCanvasTkAgg(fig, master=parent)
#         canvas.draw()
#         canvas.get_tk_widget().pack(side=side, padx=10)

#         return canvas

# if __name__ == "__main__":
#     app = AIDSSApp()
#     app.mainloop()
























# import tkinter as tk
# from tkinter import ttk
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from gui_mainboard_pages import DashboardPage, ReportsPage, DataAnalysisPage, ModelSelectionPage, DataManagementPage
# from PIL import Image, ImageTk

# class AIDSSApp(tk.Tk):
#     def __init__(self):
#         super().__init__()

#         self.title("AI-DSS")
#         self.geometry("1200x800")
#         # self.configure(bg='white')

#         bg_image_path = "../assets/canvasbg.png"
#         bg_image = Image.open(bg_image_path)
#         self.bg_render = ImageTk.PhotoImage(bg_image)  # Store the reference to avoid garbage collection

#         # Set the background label
#         background_label = tk.Label(self, image=self.bg_render)
#         background_label.place(relwidth=1, relheight=1)

#         # Top-level container frame
#         container = tk.Frame(self)
#         container.pack(expand=True)

#         # Top bar frame
#         top_bar = tk.Frame(container, height=60)
#         top_bar.pack(side='top', fill='x')

#         profile_icon = tk.Label(top_bar, text='Profile', fg='white')
#         profile_icon.pack(side='right', padx=20)

#         chatbot_icon = tk.Label(top_bar, text='Chatbot', fg='white')
#         chatbot_icon.pack(side='right', padx=20)

#         menu_icon = tk.Label(top_bar, text='Menu', fg='white')
#         menu_icon.pack(side='right', padx=20)

#         # Center container for sidebar and mainboard
#         center_container = tk.Frame(container)
#         center_container.pack(expand=True, fill='both')

#         # Sidebar frame
#         sidebar = tk.Frame(center_container, bg='white', width=200, height=600)
#         sidebar.pack(side='left', fill='y', padx=10, pady=10)
#         sidebar.pack_propagate(0)  # Prevents frame from shrinking to content size

#         self.pages = {
#             "Dashboard": DashboardPage,
#             "Reports": ReportsPage,
#             "Data Analysis": DataAnalysisPage,
#             "Model Selection": ModelSelectionPage,
#             "Data Management": DataManagementPage
#         }

#         buttons = ["Dashboard", "Reports", "Data Analysis", "Model Selection", "Data Management"]

#         for btn_text in buttons:
#             btn = tk.Button(sidebar, text=btn_text, fg='black', relief='flat',
#                             command=lambda name=btn_text: self.show_page(name))
#             btn.pack(fill='x', pady=35)

#         # Create a canvas and scrollbar for the main board
#         canvas = tk.Canvas(center_container, bg='white', width=800, height=600)
#         canvas.pack(side='right', expand=True, fill='both', padx=10, pady=10)

#         scrollbar = ttk.Scrollbar(center_container, orient='vertical', command=canvas.yview)
#         scrollbar.pack(side='right', fill='y')

#         canvas.configure(yscrollcommand=scrollbar.set)
#         canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

#         self.main_board = tk.Frame(canvas, bg='white', width=800, height=600)
#         self.main_board.pack(expand=True, fill='both')
#         self.main_board.pack_propagate(0)  # Prevents frame from shrinking to content size

#         canvas.create_window((0, 0), window=self.main_board, anchor='nw')

#         self.frames = {}
#         for name, Page in self.pages.items():
#             frame = Page(parent=self.main_board, controller=self)
#             self.frames[name] = frame
#             frame.grid(row=0, column=0, sticky="nsew")

#         self.show_page("Dashboard")

#     def show_page(self, page_name):
#         frame = self.frames[page_name]
#         frame.tkraise()

# if __name__ == "__main__":
#     app = AIDSSApp()
#     app.mainloop()




























import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from gui_mainboard_pages import DashboardPage, ReportsPage, DataAnalysisPage, ModelSelectionPage, DataManagementPage

class AIDSSApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("AI-DSS")
        self.geometry("1200x800")
        self.configure(bg='white')

        # Load background image
        bg_image_path = "../assets/canvasbg.png"
        bg_image = Image.open(bg_image_path)
        self.bg_render = ImageTk.PhotoImage(bg_image)

        # Set the background label
        background_label = tk.Label(self, image=self.bg_render)
        background_label.place(relwidth=1, relheight=1)

        # Top-level container frame
        container = tk.Frame(self)
        container.pack(expand=True)

        # Top bar frame
        top_bar = tk.Frame(container, height=60, bg='white')
        top_bar.pack(side='top', fill='x')

        # Load logo image
        logo_path = "../assets/SWIT.png"
        logo_image = Image.open(logo_path)
        logo_render = ImageTk.PhotoImage(logo_image)

        # Create logo label
        logo_label = tk.Label(top_bar, image=logo_render, bg='white')
        logo_label.image = logo_render  # Keep a reference
        logo_label.pack(pady=10)  # Add padding to center vertically

        profile_icon = tk.Label(top_bar, text='Profile', fg='#102f61', bg='white')
        profile_icon.pack(side='right', padx=20)

        chatbot_icon = tk.Label(top_bar, text='Chatbot', fg='#102f61', bg='white')
        chatbot_icon.pack(side='right', padx=20)

        menu_icon = tk.Label(top_bar, text='Menu', fg='#102f61', bg='white')
        menu_icon.pack(side='right', padx=20)

        # Center container for sidebar and mainboard
        center_container = tk.Frame(container)
        center_container.pack(expand=True, fill='both')

        # Sidebar frame
        sidebar = tk.Frame(center_container, bg='white', width=200, height=600)
        sidebar.pack(side='left', fill='y', padx=10, pady=10)
        sidebar.pack_propagate(0)  # Prevents frame from shrinking to content size

        self.pages = {
            "Dashboard": DashboardPage,
            "Reports": ReportsPage,
            "Data Analysis": DataAnalysisPage,
            "Model Selection": ModelSelectionPage,
            "Data Management": DataManagementPage
        }

        buttons = ["Dashboard", "Reports", "Data Analysis", "Model Selection", "Data Management"]

        for btn_text in buttons:
            btn = tk.Button(sidebar, text=btn_text, fg='black', relief='flat',
                            command=lambda name=btn_text: self.show_page(name))
            btn.pack(fill='x', pady=35)

        # Create a canvas and scrollbar for the main board
        canvas = tk.Canvas(center_container, bg='white', width=800, height=600)
        canvas.pack(side='right', expand=True, fill='both', padx=10, pady=10)

        scrollbar = ttk.Scrollbar(center_container, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        self.main_board = tk.Frame(canvas, bg='white', width=800, height=600)
        self.main_board.pack(expand=True, fill='both')
        self.main_board.pack_propagate(0)  # Prevents frame from shrinking to content size

        canvas.create_window((0, 0), window=self.main_board, anchor='nw')

        self.frames = {}
        for name, Page in self.pages.items():
            frame = Page(parent=self.main_board, controller=self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_page("Dashboard")

    def show_page(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

if __name__ == "__main__":
    app = AIDSSApp()
    app.mainloop()
