import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import time

class UI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("23110280_HuynhThanhNhan_HW2")
        self.window.geometry("1500x800")
        self.window.configure(bg='#f0f0f0')
        
        self.original_image = None
        self.processed_image = None
        self.photo_original = None
        self.photo_processed = None
        
        # Lưu trữ filtered channels để hiển thị sau
        self.last_filtered_channels = None
        self.last_filtered_img = None
        
        self.selected_transform = tk.StringVar(value="negative")
        self.selected_domain = tk.StringVar(value="Miền không gian")
        
        # Tạo button frame trước để đảm bảo nó luôn ở bottom
        button_frame = tk.Frame(self.window, bg='#f0f0f0')
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Main container với padding bottom đủ lớn để không che button
        main_container = tk.Frame(self.window, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 80))
        
        left_panel = tk.Frame(main_container, bg='#f0f0f0')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Sử dụng grid để chia đều không gian cho 2 ảnh
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_rowconfigure(3, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        tk.Label(left_panel, text="Original Image", font=('Arial', 10, 'bold'), bg='#f0f0f0').grid(row=0, column=0, sticky='ew', pady=(0, 5))
        self.original_label = tk.Label(left_panel, bg='black', relief=tk.SUNKEN, bd=2)
        self.original_label.grid(row=1, column=0, sticky='nsew', pady=5)
        
        self.processed_title_label = tk.Label(left_panel, text="Negative image", 
                                              font=('Arial', 10, 'bold'), bg='#f0f0f0')
        self.processed_title_label.grid(row=2, column=0, sticky='ew', pady=(10, 5))
        self.processed_label = tk.Label(left_panel, bg='black', relief=tk.SUNKEN, bd=2)
        self.processed_label.grid(row=3, column=0, sticky='nsew', pady=5)
        
        right_panel = tk.Frame(main_container, bg='#f0f0f0', width=580)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        canvas = tk.Canvas(right_panel, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0', width=560)
        
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Đảm bảo window trong canvas có width cố định
            canvas.itemconfig(canvas.find_all()[0], width=560)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=560)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Tiêu đề chính
        title_label = tk.Label(scrollable_frame, text="Công cụ biến đổi", 
                              font=('Arial', 14, 'bold'), bg='#00bcd4', fg='white', pady=8)
        title_label.pack(fill=tk.X, pady=(0, 10))
        
        # Menu chọn miền
        menu_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
        menu_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        tk.Label(menu_frame, text="Chọn miền:", font=('Arial', 11, 'bold'), bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        
        # Tạo OptionMenu với giá trị tiếng Việt
        domain_options = ["Miền không gian", "Miền tần số"]
        domain_menu = tk.OptionMenu(menu_frame, self.selected_domain, *domain_options, 
                                    command=self.on_domain_change)
        domain_menu.config(font=('Arial', 10, 'bold'), bg='#2196f3', fg='white', 
                          activebackground='#1976d2', activeforeground='white', width=20)
        domain_menu.pack(side=tk.LEFT, padx=5)
        
        # Set giá trị mặc định
        self.selected_domain.set("Miền không gian")
        
        # === FRAME MIỀN KHÔNG GIAN ===
        self.spatial_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
        self.spatial_frame.pack(fill=tk.X, pady=(0, 10))
        
        spatial_title = tk.Label(self.spatial_frame, text="📐 MIỀN KHÔNG GIAN (Spatial Domain)", 
                               font=('Arial', 12, 'bold'), bg='#4caf50', fg='white', pady=8)
        spatial_title.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.create_negative_section(self.spatial_frame)
        self.create_log_section(self.spatial_frame)
        self.create_piecewise_section(self.spatial_frame)
        self.create_gamma_section(self.spatial_frame)
        self.create_average_filter_section(self.spatial_frame)
        self.create_gaussian_filter_section(self.spatial_frame)
        self.create_median_filter_section(self.spatial_frame)
        self.create_min_filter_section(self.spatial_frame)
        self.create_max_filter_section(self.spatial_frame)
        self.create_midpoint_filter_section(self.spatial_frame)
        self.create_histogram_section(self.spatial_frame)
        self.create_sharpening_section(self.spatial_frame)
        self.create_line_detection_section(self.spatial_frame)
        self.create_adaptive_thresholding_section(self.spatial_frame)
        self.create_optimum_thresholding_section(self.spatial_frame)
        self.create_binary_conversion_section(self.spatial_frame)
        self.create_binary_analysis_section(self.spatial_frame)
        self.create_dilation_section(self.spatial_frame)
        self.create_erosion_section(self.spatial_frame)
        self.create_opening_section(self.spatial_frame)
        self.create_closing_section(self.spatial_frame)
        self.create_morphological_filtering_section(self.spatial_frame)
        
        # === FRAME MIỀN TẦN SỐ ===
        self.frequency_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
        self.frequency_frame.pack(fill=tk.X, pady=(0, 10))
        
        frequency_title = tk.Label(self.frequency_frame, text="📊 MIỀN TẦN SỐ (Frequency Domain)", 
                                  font=('Arial', 12, 'bold'), bg='#9c27b0', fg='white', pady=8)
        frequency_title.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.create_fourier_section(self.frequency_frame)
        self.create_frequency_filter_section(self.frequency_frame)
        
        # Khởi tạo hiển thị
        self.on_domain_change()
        
        self.create_bottom_buttons(button_frame)
    
    def on_domain_change(self, *args):
        """Ẩn/hiện frame tương ứng khi chọn miền"""
        domain = self.selected_domain.get()
        
        if domain == "Miền không gian":
            self.spatial_frame.pack(fill=tk.X, pady=(0, 10))
            self.frequency_frame.pack_forget()
        else:  # Miền tần số
            self.frequency_frame.pack(fill=tk.X, pady=(0, 10))
            self.spatial_frame.pack_forget()
        
    def create_negative_section(self, parent):
        frame = tk.Frame(parent, bg='#e3f2fd', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Negative image", variable=self.selected_transform,
                           value="negative", bg='#90caf9', fg='black',
                           font=('Arial', 10, 'bold'), selectcolor='#64b5f6',
                           command=self.apply_transformation)
        rb.pack(pady=10, padx=10, fill=tk.X)
    
    def create_log_section(self, parent):
        frame = tk.Frame(parent, bg='#2196f3', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Biến đổi Log", variable=self.selected_transform,
                           value="log", bg='#2196f3', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#64b5f6',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        c_frame = tk.Frame(frame, bg='#2196f3')
        c_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(c_frame, text="Hệ số C", bg='#2196f3', fg='white').pack(side=tk.LEFT)
        self.log_c_slider = tk.Scale(c_frame, from_=0.1, to=50, resolution=0.1, 
                                     orient=tk.HORIZONTAL, bg='#64b5f6', length=200,
                                     command=lambda x: self.apply_transformation())
        self.log_c_slider.set(1.0)
        self.log_c_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_piecewise_section(self, parent):
        frame = tk.Frame(parent, bg='#9c27b0', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Biến đổi Piecewise-Linear", variable=self.selected_transform,
                           value="piecewise", bg='#9c27b0', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#ba68c8',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        high_frame = tk.Frame(frame, bg='#9c27b0')
        high_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(high_frame, text="Hệ số Cao", bg='#9c27b0', fg='white').pack(side=tk.LEFT)
        self.piecewise_high = tk.Scale(high_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                      bg='#ba68c8', length=200,
                                      command=lambda x: self.apply_transformation())
        self.piecewise_high.set(200)
        self.piecewise_high.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        low_frame = tk.Frame(frame, bg='#9c27b0')
        low_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(low_frame, text="Hệ số Thấp", bg='#9c27b0', fg='white').pack(side=tk.LEFT)
        self.piecewise_low = tk.Scale(low_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     bg='#ba68c8', length=200,
                                     command=lambda x: self.apply_transformation())
        self.piecewise_low.set(50)
        self.piecewise_low.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_gamma_section(self, parent):
        frame = tk.Frame(parent, bg='#ff6f00', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Biến đổi Gamma", variable=self.selected_transform,
                           value="gamma", bg='#ff6f00', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#ff9800',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        c_frame = tk.Frame(frame, bg='#ff6f00')
        c_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(c_frame, text="Hệ số C", bg='#ff6f00', fg='white').pack(side=tk.LEFT)
        self.gamma_c = tk.Scale(c_frame, from_=0.1, to=10, resolution=0.1,
                               orient=tk.HORIZONTAL, bg='#ff9800', length=200,
                               command=lambda x: self.apply_transformation())
        self.gamma_c.set(1.0)
        self.gamma_c.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        gamma_frame = tk.Frame(frame, bg='#ff6f00')
        gamma_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(gamma_frame, text="Gamma", bg='#ff6f00', fg='white').pack(side=tk.LEFT)
        self.gamma_value = tk.Scale(gamma_frame, from_=0.1, to=5, resolution=0.1,
                                    orient=tk.HORIZONTAL, bg='#ff9800', length=200,
                                    command=lambda x: self.apply_transformation())
        self.gamma_value.set(1.0)
        self.gamma_value.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_average_filter_section(self, parent):
        frame = tk.Frame(parent, bg='#e91e63', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Làm trơn ảnh (lọc trung bình)", variable=self.selected_transform,
                           value="average", bg='#e91e63', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#f06292',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        size_frame = tk.Frame(frame, bg='#e91e63')
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(size_frame, text="Vùng lượng lập", bg='#e91e63', fg='white').pack(side=tk.LEFT)
        self.avg_kernel = tk.Scale(size_frame, from_=3, to=15, resolution=2,
                                  orient=tk.HORIZONTAL, bg='#f06292', length=200,
                                  command=lambda x: self.apply_transformation())
        self.avg_kernel.set(3)
        self.avg_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_gaussian_filter_section(self, parent):
        frame = tk.Frame(parent, bg='#ff6f00', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Làm trơn ảnh (lọc Gauss)", variable=self.selected_transform,
                           value="gaussian", bg='#ff6f00', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#ff9800',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        kernel_frame = tk.Frame(frame, bg='#ff6f00')
        kernel_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(kernel_frame, text="Kích trước lọc", bg='#ff6f00', fg='white').pack(side=tk.LEFT)
        self.gauss_kernel = tk.Scale(kernel_frame, from_=3, to=15, resolution=2,
                                    orient=tk.HORIZONTAL, bg='#ff9800', length=200,
                                    command=lambda x: self.apply_transformation())
        self.gauss_kernel.set(3)
        self.gauss_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        sigma_frame = tk.Frame(frame, bg='#ff6f00')
        sigma_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(sigma_frame, text="Hệ số Sigma", bg='#ff6f00', fg='white').pack(side=tk.LEFT)
        self.gauss_sigma = tk.Scale(sigma_frame, from_=0.1, to=10, resolution=0.1,
                                   orient=tk.HORIZONTAL, bg='#ff9800', length=200,
                                   command=lambda x: self.apply_transformation())
        self.gauss_sigma.set(1.0)
        self.gauss_sigma.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_median_filter_section(self, parent):
        frame = tk.Frame(parent, bg='#e91e63', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Làm trơn ảnh (lọc trung vị)", variable=self.selected_transform,
                           value="median", bg='#e91e63', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#f06292',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        size_frame = tk.Frame(frame, bg='#e91e63')
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(size_frame, text="Vùng lượng lập", bg='#e91e63', fg='white').pack(side=tk.LEFT)
        self.median_kernel = tk.Scale(size_frame, from_=3, to=15, resolution=2,
                                     orient=tk.HORIZONTAL, bg='#f06292', length=200,
                                     command=lambda x: self.apply_transformation())
        self.median_kernel.set(3)
        self.median_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_min_filter_section(self, parent):
        frame = tk.Frame(parent, bg='#4caf50', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Lọc Min", variable=self.selected_transform,
                           value="min", bg='#4caf50', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#81c784',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        size_frame = tk.Frame(frame, bg='#4caf50')
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(size_frame, text="Kích thước lọc", bg='#4caf50', fg='white').pack(side=tk.LEFT)
        self.min_kernel = tk.Scale(size_frame, from_=3, to=15, resolution=2,
                                   orient=tk.HORIZONTAL, bg='#81c784', length=200,
                                   command=lambda x: self.apply_transformation())
        self.min_kernel.set(3)
        self.min_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_max_filter_section(self, parent):
        frame = tk.Frame(parent, bg='#f44336', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Lọc Max", variable=self.selected_transform,
                           value="max", bg='#f44336', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#e57373',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        size_frame = tk.Frame(frame, bg='#f44336')
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(size_frame, text="Kích thước lọc", bg='#f44336', fg='white').pack(side=tk.LEFT)
        self.max_kernel = tk.Scale(size_frame, from_=3, to=15, resolution=2,
                                   orient=tk.HORIZONTAL, bg='#e57373', length=200,
                                   command=lambda x: self.apply_transformation())
        self.max_kernel.set(3)
        self.max_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_midpoint_filter_section(self, parent):
        frame = tk.Frame(parent, bg='#673ab7', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Lọc Midpoint", variable=self.selected_transform,
                           value="midpoint", bg='#673ab7', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#9575cd',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        size_frame = tk.Frame(frame, bg='#673ab7')
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(size_frame, text="Kích thước lọc", bg='#673ab7', fg='white').pack(side=tk.LEFT)
        self.midpoint_kernel = tk.Scale(size_frame, from_=3, to=15, resolution=2,
                                       orient=tk.HORIZONTAL, bg='#9575cd', length=200,
                                       command=lambda x: self.apply_transformation())
        self.midpoint_kernel.set(3)
        self.midpoint_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_histogram_section(self, parent):
        frame = tk.Frame(parent, bg='#fbc02d', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Cân bằng sáng dùng Histogram", variable=self.selected_transform,
                           value="histogram", bg='#fbc02d', fg='black',
                           font=('Arial', 10, 'bold'), selectcolor='#fff176',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        value_frame = tk.Frame(frame, bg='#fbc02d')
        value_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(value_frame, text="Giá trị", bg='#fbc02d', fg='black').pack(side=tk.LEFT)
        self.hist_value = tk.Scale(value_frame, from_=0, to=1, resolution=0.1,
                                  orient=tk.HORIZONTAL, bg='#fff176', length=200,
                                  command=lambda x: self.apply_transformation())
        self.hist_value.set(1.0)
        self.hist_value.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_sharpening_section(self, parent):
        frame = tk.Frame(parent, bg='#00acc1', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Image Sharpening Workflow", variable=self.selected_transform,
                           value="sharpening", bg='#00acc1', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#4dd0e1',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        gamma_frame = tk.Frame(frame, bg='#00acc1')
        gamma_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(gamma_frame, text="Power-law Gamma", bg='#00acc1', fg='white').pack(side=tk.LEFT)
        self.sharpening_gamma = tk.Scale(gamma_frame, from_=0.1, to=3.0, resolution=0.1,
                                         orient=tk.HORIZONTAL, bg='#4dd0e1', length=200,
                                         command=lambda x: self.apply_transformation())
        self.sharpening_gamma.set(0.5)
        self.sharpening_gamma.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_line_detection_section(self, parent):
        frame = tk.Frame(parent, bg='#ff9800', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Line Detection (Laplacian)", variable=self.selected_transform,
                           value="line_detection", bg='#ff9800', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#ffb74d',
                           command=self.apply_line_detection)
        rb.pack(anchor='w', padx=10, pady=5)
    
    def create_adaptive_thresholding_section(self, parent):
        frame = tk.Frame(parent, bg='#607d8b', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Adaptive Thresholding", variable=self.selected_transform,
                           value="adaptive_threshold", bg='#607d8b', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#90a4ae',
                           command=self.apply_adaptive_thresholding)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        # Parameters frame
        params_frame = tk.Frame(frame, bg='#607d8b')
        params_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        
        # Threshold value
        tk.Label(params_frame, text="Threshold Value:", bg='#607d8b', fg='white').pack(side=tk.LEFT, padx=5)
        self.adaptive_thresh_val = tk.Scale(params_frame, from_=0, to=255, resolution=1,
                                             orient=tk.HORIZONTAL, bg='#90a4ae', length=150,
                                             command=lambda x: self.apply_adaptive_thresholding() if self.selected_transform.get() == "adaptive_threshold" else None)
        self.adaptive_thresh_val.set(132)
        self.adaptive_thresh_val.pack(side=tk.LEFT, padx=5)
        
        # Block size
        tk.Label(params_frame, text="Block Size:", bg='#607d8b', fg='white').pack(side=tk.LEFT, padx=5)
        self.adaptive_block_size = tk.Scale(params_frame, from_=3, to=31, resolution=2,
                                            orient=tk.HORIZONTAL, bg='#90a4ae', length=100,
                                            command=lambda x: self.apply_adaptive_thresholding() if self.selected_transform.get() == "adaptive_threshold" else None)
        self.adaptive_block_size.set(11)
        self.adaptive_block_size.pack(side=tk.LEFT, padx=5)
        
        # C value
        tk.Label(params_frame, text="C:", bg='#607d8b', fg='white').pack(side=tk.LEFT, padx=5)
        self.adaptive_c = tk.Scale(params_frame, from_=0, to=20, resolution=1,
                                   orient=tk.HORIZONTAL, bg='#90a4ae', length=100,
                                   command=lambda x: self.apply_adaptive_thresholding() if self.selected_transform.get() == "adaptive_threshold" else None)
        self.adaptive_c.set(2)
        self.adaptive_c.pack(side=tk.LEFT, padx=5)
    
    def create_optimum_thresholding_section(self, parent):
        frame = tk.Frame(parent, bg='#795548', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Optimum Thresholding (Local)", variable=self.selected_transform,
                           value="optimum_threshold", bg='#795548', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#a1887f',
                           command=self.apply_optimum_thresholding)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        # Parameters frame
        params_frame = tk.Frame(frame, bg='#795548')
        params_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        
        # Note: Otsu threshold is calculated automatically, no need for threshold value slider
        tk.Label(params_frame, text="(Otsu threshold tự động)", bg='#795548', fg='white', 
                font=('Arial', 9, 'italic')).pack(side=tk.LEFT, padx=5)
        
        # Block size
        tk.Label(params_frame, text="Block Size:", bg='#795548', fg='white').pack(side=tk.LEFT, padx=5)
        self.optimum_block_size = tk.Scale(params_frame, from_=3, to=31, resolution=2,
                                           orient=tk.HORIZONTAL, bg='#a1887f', length=100,
                                           command=lambda x: self.apply_optimum_thresholding() if self.selected_transform.get() == "optimum_threshold" else None)
        self.optimum_block_size.set(7)
        self.optimum_block_size.pack(side=tk.LEFT, padx=5)
        
        # C value
        tk.Label(params_frame, text="C:", bg='#795548', fg='white').pack(side=tk.LEFT, padx=5)
        self.optimum_c = tk.Scale(params_frame, from_=0, to=20, resolution=1,
                                  orient=tk.HORIZONTAL, bg='#a1887f', length=100,
                                  command=lambda x: self.apply_optimum_thresholding() if self.selected_transform.get() == "optimum_threshold" else None)
        self.optimum_c.set(2)
        self.optimum_c.pack(side=tk.LEFT, padx=5)
    
    def create_binary_conversion_section(self, parent):
        frame = tk.Frame(parent, bg='#795548', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Chuyển đổi ảnh nhị phân", variable=self.selected_transform,
                           value="binary", bg='#795548', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#a1887f',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        threshold_frame = tk.Frame(frame, bg='#795548')
        threshold_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(threshold_frame, text="Ngưỡng", bg='#795548', fg='white').pack(side=tk.LEFT)
        self.binary_threshold = tk.Scale(threshold_frame, from_=0, to=255, resolution=1,
                                         orient=tk.HORIZONTAL, bg='#a1887f', length=200,
                                         command=lambda x: self.apply_transformation())
        self.binary_threshold.set(127)
        self.binary_threshold.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_binary_analysis_section(self, parent):
        frame = tk.Frame(parent, bg='#607d8b', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        tk.Label(frame, text="Phân tích ảnh nhị phân", 
                bg='#607d8b', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)
        
        tk.Button(frame, text="Phân tích đối tượng trong ảnh nhị phân", 
                 command=self.analyze_binary_image_action,
                 bg='#78909c', fg='white', font=('Arial', 9, 'bold'),
                 width=35, height=2).pack(pady=10, padx=10)
    
    def create_dilation_section(self, parent):
        frame = tk.Frame(parent, bg='#e91e63', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Dilation (Giãn nở)", variable=self.selected_transform,
                           value="dilation", bg='#e91e63', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#f06292',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        kernel_frame = tk.Frame(frame, bg='#e91e63')
        kernel_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(kernel_frame, text="Kích thước kernel", bg='#e91e63', fg='white').pack(side=tk.LEFT)
        self.dilation_kernel = tk.Scale(kernel_frame, from_=3, to=15, resolution=2,
                                       orient=tk.HORIZONTAL, bg='#f06292', length=200,
                                       command=lambda x: self.apply_transformation())
        self.dilation_kernel.set(3)
        self.dilation_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        iter_frame = tk.Frame(frame, bg='#e91e63')
        iter_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(iter_frame, text="Số lần lặp", bg='#e91e63', fg='white').pack(side=tk.LEFT)
        self.dilation_iterations = tk.Scale(iter_frame, from_=1, to=10, resolution=1,
                                           orient=tk.HORIZONTAL, bg='#f06292', length=200,
                                           command=lambda x: self.apply_transformation())
        self.dilation_iterations.set(1)
        self.dilation_iterations.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_erosion_section(self, parent):
        frame = tk.Frame(parent, bg='#3f51b5', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Erosion (Bào mòn)", variable=self.selected_transform,
                           value="erosion", bg='#3f51b5', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#7986cb',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        kernel_frame = tk.Frame(frame, bg='#3f51b5')
        kernel_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(kernel_frame, text="Kích thước kernel", bg='#3f51b5', fg='white').pack(side=tk.LEFT)
        self.erosion_kernel = tk.Scale(kernel_frame, from_=3, to=15, resolution=2,
                                       orient=tk.HORIZONTAL, bg='#7986cb', length=200,
                                       command=lambda x: self.apply_transformation())
        self.erosion_kernel.set(3)
        self.erosion_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        iter_frame = tk.Frame(frame, bg='#3f51b5')
        iter_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(iter_frame, text="Số lần lặp", bg='#3f51b5', fg='white').pack(side=tk.LEFT)
        self.erosion_iterations = tk.Scale(iter_frame, from_=1, to=10, resolution=1,
                                           orient=tk.HORIZONTAL, bg='#7986cb', length=200,
                                           command=lambda x: self.apply_transformation())
        self.erosion_iterations.set(1)
        self.erosion_iterations.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_opening_section(self, parent):
        frame = tk.Frame(parent, bg='#009688', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Opening (Mở)", variable=self.selected_transform,
                           value="opening", bg='#009688', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#4db6ac',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        kernel_frame = tk.Frame(frame, bg='#009688')
        kernel_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(kernel_frame, text="Kích thước kernel", bg='#009688', fg='white').pack(side=tk.LEFT)
        self.opening_kernel = tk.Scale(kernel_frame, from_=3, to=15, resolution=2,
                                      orient=tk.HORIZONTAL, bg='#4db6ac', length=200,
                                      command=lambda x: self.apply_transformation())
        self.opening_kernel.set(3)
        self.opening_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_closing_section(self, parent):
        frame = tk.Frame(parent, bg='#ff5722', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Closing (Đóng)", variable=self.selected_transform,
                           value="closing", bg='#ff5722', fg='white',
                           font=('Arial', 10, 'bold'), selectcolor='#ff8a65',
                           command=self.apply_transformation)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        kernel_frame = tk.Frame(frame, bg='#ff5722')
        kernel_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(kernel_frame, text="Kích thước kernel", bg='#ff5722', fg='white').pack(side=tk.LEFT)
        self.closing_kernel = tk.Scale(kernel_frame, from_=3, to=15, resolution=2,
                                      orient=tk.HORIZONTAL, bg='#ff8a65', length=200,
                                      command=lambda x: self.apply_transformation())
        self.closing_kernel.set(3)
        self.closing_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_morphological_filtering_section(self, parent):
        """Section cho quy trình lọc hình thái học đầy đủ"""
        frame = tk.Frame(parent, bg='#795548', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        tk.Label(frame, text="Lọc hình thái học (Opening + Closing)", 
                bg='#795548', fg='white', font=('Arial', 10, 'bold')).pack(pady=(5, 0))
        
        kernel_frame = tk.Frame(frame, bg='#795548')
        kernel_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(kernel_frame, text="Kích thước kernel", bg='#795548', fg='white').pack(side=tk.LEFT)
        self.morph_kernel = tk.Scale(kernel_frame, from_=3, to=15, resolution=2,
                                     orient=tk.HORIZONTAL, bg='#a1887f', length=200)
        self.morph_kernel.set(3)
        self.morph_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        tk.Button(frame, text="Thực hiện lọc hình thái học", 
                 command=self.apply_morphological_filtering,
                 bg='#5d4037', fg='white', font=('Arial', 9, 'bold'),
                 relief=tk.RAISED, bd=2).pack(pady=(0, 5), padx=10, fill=tk.X)
    
    def create_fourier_section(self, parent):
        frame = tk.Frame(parent, bg='#9c27b0', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        tk.Label(frame, text="Biến đổi Fourier (Miền tần số)", 
                bg='#9c27b0', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)
        
        tk.Button(frame, text="Chuyển sang miền tần số", 
                 command=self.apply_fourier_transform,
                 bg='#ba68c8', fg='white', font=('Arial', 9, 'bold'),
                 width=25, height=2).pack(pady=10, padx=10)
    
    def create_frequency_filter_section(self, parent):
        # Khởi tạo biến chung cho việc chọn filter
        self.filter_type = tk.StringVar(value="gaussian")
        
        # Tạo các frame riêng cho từng loại filter
        self.create_gaussian_lowpass_section(parent)
        self.create_gaussian_highpass_section(parent)
        self.create_butterworth_lowpass_section(parent)
        self.create_butterworth_highpass_section(parent)
        self.create_ideal_lowpass_section(parent)
        self.create_ideal_highpass_section(parent)
        
        # Section so sánh Gaussian giữa 2 miền
        self.create_gaussian_comparison_section(parent)
    
    def apply_frequency_filter_if_selected(self, filter_value):
        """Chỉ áp dụng filter nếu filter này đang được chọn"""
        if self.filter_type.get() == filter_value:
            self.apply_frequency_filter()
    
    def create_gaussian_lowpass_section(self, parent):
        frame = tk.Frame(parent, bg='#673ab7', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Gaussian Low-pass Filter", 
                           variable=self.filter_type, value="gaussian",
                           bg='#673ab7', fg='white', font=('Arial', 10, 'bold'),
                           selectcolor='#9575cd', command=self.apply_frequency_filter)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        # Tham số D0 (cutoff frequency)
        d0_frame = tk.Frame(frame, bg='#673ab7')
        d0_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(d0_frame, text="D0 (Cutoff):", bg='#673ab7', fg='white').pack(side=tk.LEFT)
        self.filter_d0_gaussian = tk.Scale(d0_frame, from_=10, to=200, resolution=5,
                                           orient=tk.HORIZONTAL, bg='#9575cd', length=200,
                                           command=lambda x: self.apply_frequency_filter_if_selected("gaussian"))
        self.filter_d0_gaussian.set(40)
        self.filter_d0_gaussian.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_gaussian_highpass_section(self, parent):
        frame = tk.Frame(parent, bg='#2196f3', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Gaussian High-pass Filter", 
                           variable=self.filter_type, value="gaussian_highpass",
                           bg='#2196f3', fg='white', font=('Arial', 10, 'bold'),
                           selectcolor='#64b5f6', command=self.apply_frequency_filter)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        # Tham số D0 (cutoff frequency)
        d0_frame = tk.Frame(frame, bg='#2196f3')
        d0_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(d0_frame, text="D0 (Cutoff):", bg='#2196f3', fg='white').pack(side=tk.LEFT)
        self.filter_d0_gaussian_highpass = tk.Scale(d0_frame, from_=10, to=200, resolution=5,
                                                     orient=tk.HORIZONTAL, bg='#64b5f6', length=200,
                                                     command=lambda x: self.apply_frequency_filter_if_selected("gaussian_highpass"))
        self.filter_d0_gaussian_highpass.set(40)
        self.filter_d0_gaussian_highpass.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_butterworth_lowpass_section(self, parent):
        frame = tk.Frame(parent, bg='#9c27b0', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Butterworth Low-pass Filter", 
                           variable=self.filter_type, value="butterworth",
                           bg='#9c27b0', fg='white', font=('Arial', 10, 'bold'),
                           selectcolor='#ba68c8', command=self.apply_frequency_filter)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        # Tham số D0 (cutoff frequency)
        d0_frame = tk.Frame(frame, bg='#9c27b0')
        d0_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(d0_frame, text="D0 (Cutoff):", bg='#9c27b0', fg='white').pack(side=tk.LEFT)
        self.filter_d0_butterworth = tk.Scale(d0_frame, from_=10, to=200, resolution=5,
                                              orient=tk.HORIZONTAL, bg='#ba68c8', length=200,
                                              command=lambda x: self.apply_frequency_filter_if_selected("butterworth"))
        self.filter_d0_butterworth.set(40)
        self.filter_d0_butterworth.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Tham số n (bậc)
        n_frame = tk.Frame(frame, bg='#9c27b0')
        n_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(n_frame, text="n (Bậc):", bg='#9c27b0', fg='white').pack(side=tk.LEFT)
        self.filter_n = tk.Scale(n_frame, from_=1, to=10, resolution=1,
                                orient=tk.HORIZONTAL, bg='#ba68c8', length=200,
                                command=lambda x: self.apply_frequency_filter_if_selected("butterworth"))
        self.filter_n.set(2)
        self.filter_n.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_butterworth_highpass_section(self, parent):
        frame = tk.Frame(parent, bg='#795548', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Butterworth High-pass Filter", 
                           variable=self.filter_type, value="butterworth_highpass",
                           bg='#795548', fg='white', font=('Arial', 10, 'bold'),
                           selectcolor='#a1887f', command=self.apply_frequency_filter)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        # Tham số D0 (cutoff frequency)
        d0_frame = tk.Frame(frame, bg='#795548')
        d0_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(d0_frame, text="D0 (Cutoff):", bg='#795548', fg='white').pack(side=tk.LEFT)
        self.filter_d0_butterworth_highpass = tk.Scale(d0_frame, from_=10, to=200, resolution=5,
                                                       orient=tk.HORIZONTAL, bg='#a1887f', length=200,
                                                       command=lambda x: self.apply_frequency_filter_if_selected("butterworth_highpass"))
        self.filter_d0_butterworth_highpass.set(40)
        self.filter_d0_butterworth_highpass.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Tham số n (bậc)
        n_frame = tk.Frame(frame, bg='#795548')
        n_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        tk.Label(n_frame, text="n (Bậc):", bg='#795548', fg='white').pack(side=tk.LEFT)
        self.filter_n_butterworth_highpass = tk.Scale(n_frame, from_=1, to=10, resolution=1,
                                                      orient=tk.HORIZONTAL, bg='#a1887f', length=200,
                                                      command=lambda x: self.apply_frequency_filter_if_selected("butterworth_highpass"))
        self.filter_n_butterworth_highpass.set(2)
        self.filter_n_butterworth_highpass.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_ideal_lowpass_section(self, parent):
        frame = tk.Frame(parent, bg='#e91e63', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Ideal Low-pass Filter", 
                           variable=self.filter_type, value="ideal",
                           bg='#e91e63', fg='white', font=('Arial', 10, 'bold'),
                           selectcolor='#f06292', command=self.apply_frequency_filter)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        # Tham số D0 (cutoff frequency)
        d0_frame = tk.Frame(frame, bg='#e91e63')
        d0_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(d0_frame, text="D0 (Cutoff):", bg='#e91e63', fg='white').pack(side=tk.LEFT)
        self.filter_d0_ideal = tk.Scale(d0_frame, from_=10, to=200, resolution=5,
                                       orient=tk.HORIZONTAL, bg='#f06292', length=200,
                                       command=lambda x: self.apply_frequency_filter_if_selected("ideal"))
        self.filter_d0_ideal.set(40)
        self.filter_d0_ideal.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_ideal_highpass_section(self, parent):
        frame = tk.Frame(parent, bg='#ff5722', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        rb = tk.Radiobutton(frame, text="Ideal High-pass Filter", 
                           variable=self.filter_type, value="ideal_highpass",
                           bg='#ff5722', fg='white', font=('Arial', 10, 'bold'),
                           selectcolor='#ff8a65', command=self.apply_frequency_filter)
        rb.pack(anchor='w', padx=10, pady=(5, 0))
        
        # Tham số D0 (cutoff frequency)
        d0_frame = tk.Frame(frame, bg='#ff5722')
        d0_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(d0_frame, text="D0 (Cutoff):", bg='#ff5722', fg='white').pack(side=tk.LEFT)
        self.filter_d0_ideal_highpass = tk.Scale(d0_frame, from_=10, to=200, resolution=5,
                                                  orient=tk.HORIZONTAL, bg='#ff8a65', length=200,
                                                  command=lambda x: self.apply_frequency_filter_if_selected("ideal_highpass"))
        self.filter_d0_ideal_highpass.set(40)
        self.filter_d0_ideal_highpass.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_gaussian_comparison_section(self, parent):
        """Tạo section để so sánh Gaussian filter giữa miền không gian và miền tần số"""
        frame = tk.Frame(parent, bg='#607d8b', relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        tk.Label(frame, text="So sánh Gaussian Filter", 
                bg='#607d8b', fg='white', font=('Arial', 11, 'bold')).pack(pady=5)
        
        # Tham số cho miền không gian
        spatial_frame = tk.Frame(frame, bg='#607d8b')
        spatial_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(spatial_frame, text="Miền không gian:", bg='#607d8b', fg='white', 
                font=('Arial', 9, 'bold')).pack(anchor='w')
        
        kernel_frame = tk.Frame(spatial_frame, bg='#607d8b')
        kernel_frame.pack(fill=tk.X, pady=2)
        tk.Label(kernel_frame, text="Kích thước:", bg='#607d8b', fg='white').pack(side=tk.LEFT)
        self.compare_gauss_kernel = tk.Scale(kernel_frame, from_=3, to=15, resolution=2,
                                             orient=tk.HORIZONTAL, bg='#78909c', length=200,
                                             command=lambda x: None)
        self.compare_gauss_kernel.set(5)
        self.compare_gauss_kernel.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        sigma_frame = tk.Frame(spatial_frame, bg='#607d8b')
        sigma_frame.pack(fill=tk.X, pady=2)
        tk.Label(sigma_frame, text="Sigma:", bg='#607d8b', fg='white').pack(side=tk.LEFT)
        self.compare_gauss_sigma = tk.Scale(sigma_frame, from_=0.1, to=5, resolution=0.1,
                                            orient=tk.HORIZONTAL, bg='#78909c', length=200,
                                            command=lambda x: None)
        self.compare_gauss_sigma.set(1.0)
        self.compare_gauss_sigma.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Tham số cho miền tần số
        freq_frame = tk.Frame(frame, bg='#607d8b')
        freq_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(freq_frame, text="Miền tần số:", bg='#607d8b', fg='white', 
                font=('Arial', 9, 'bold')).pack(anchor='w')
        
        d0_frame = tk.Frame(freq_frame, bg='#607d8b')
        d0_frame.pack(fill=tk.X, pady=2)
        tk.Label(d0_frame, text="D0 (Cutoff):", bg='#607d8b', fg='white').pack(side=tk.LEFT)
        self.compare_gauss_d0 = tk.Scale(d0_frame, from_=10, to=200, resolution=5,
                                         orient=tk.HORIZONTAL, bg='#78909c', length=200,
                                         command=lambda x: None)
        self.compare_gauss_d0.set(40)
        self.compare_gauss_d0.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Nút so sánh
        tk.Button(frame, text="So sánh Gaussian Filter", 
                 command=self.compare_gaussian_filters,
                 bg='#455a64', fg='white', font=('Arial', 10, 'bold'),
                 width=25, height=2).pack(pady=10, padx=10)
    
    def create_bottom_buttons(self, button_frame):
        tk.Button(button_frame, text="Chọn ảnh", command=self.select_image, 
                 bg='#4caf50', fg='white', font=('Arial', 10, 'bold'), 
                 width=12, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Cập nhật", command=self.update_image,
                 bg='#2196f3', fg='white', font=('Arial', 10, 'bold'),
                 width=12, height=2).pack(side=tk.LEFT, padx=5)
        
        # Nút hiển thị 3 kênh ảnh (chỉ hiển thị khi có filtered_channels)
        self.show_channels_button = tk.Button(
            button_frame, 
            text="Hiển thị 3 kênh ảnh", 
            command=self.show_channels_window,
            bg='#9c27b0', 
            fg='white', 
            font=('Arial', 10, 'bold'),
            width=15, 
            height=2,
            state=tk.DISABLED
        )
        self.show_channels_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Line Detection", command=self.apply_line_detection,
                 bg='#ff5722', fg='white', font=('Arial', 10, 'bold'),
                 width=15, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Lưu ra file", command=self.save_image,
                 bg='#ff9800', fg='white', font=('Arial', 10, 'bold'),
                 width=12, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Close", command=self.window.quit,
                 bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                 width=12, height=2).pack(side=tk.LEFT, padx=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            img_color = cv2.imread(file_path)
            if img_color is not None:
                self.original_image = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                
                # Reset filtered channels khi chọn ảnh mới
                self.last_filtered_channels = None
                self.last_filtered_img = None
                if hasattr(self, 'show_channels_button'):
                    self.show_channels_button.config(state=tk.DISABLED)
                
                self.show_original_image()
                self.apply_transformation()
                print("Image loaded okk!!!")
            else:
                messagebox.showerror("Error", "Could not load image")

    def show_original_image(self):
        if self.original_image is not None:
            img_resized = self.resize_for_display(self.original_image)
            image = Image.fromarray(img_resized)
            self.photo_original = ImageTk.PhotoImage(image=image)
            self.original_label.config(image=self.photo_original)
            
    def show_processed_image(self, img):
        if img is not None:
            img_resized = self.resize_for_display(img)
            image = Image.fromarray(img_resized)
            self.photo_processed = ImageTk.PhotoImage(image=image)
            self.processed_label.config(image=self.photo_processed)
    
    def show_fourier_results_window(self, input_img, spectrum_img, ifft_img, channels_magnitude=None):
        """Hiển thị ảnh trong cửa sổ mới: Input, Spectrum, IFFT, và từng kênh (nếu có)"""
        # Tạo cửa sổ mới với scroll
        fourier_window = tk.Toplevel(self.window)
        fourier_window.title("Biến đổi Fourier - Kết quả")
        fourier_window.geometry("1600x1000")
        fourier_window.configure(bg='#f0f0f0')
        
        # Container với scroll
        canvas = tk.Canvas(fourier_window, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(fourier_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Hàng 1: Input Image, Spectrum, IFFT Image
        row1 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row1.pack(fill=tk.X, pady=5)
        
        # Ảnh 1: Input Image
        frame1 = tk.Frame(row1, bg='#f0f0f0')
        frame1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame1, text="Input Image", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        label1 = tk.Label(frame1, bg='black', relief=tk.SUNKEN, bd=2)
        label1.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img1_resized = self.resize_for_display(input_img, max_width=350, max_height=350)
        image1 = Image.fromarray(img1_resized)
        photo1 = ImageTk.PhotoImage(image=image1)
        label1.config(image=photo1)
        label1.image = photo1
        
        # Ảnh 2: Spectrum
        frame2 = tk.Frame(row1, bg='#f0f0f0')
        frame2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame2, text="Spectrum (Frequency Domain)", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        label2 = tk.Label(frame2, bg='black', relief=tk.SUNKEN, bd=2)
        label2.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img2_resized = self.resize_for_display(spectrum_img, max_width=350, max_height=350)
        image2 = Image.fromarray(img2_resized)
        photo2 = ImageTk.PhotoImage(image=image2)
        label2.config(image=photo2)
        label2.image = photo2
        
        # Ảnh 3: IFFT Image
        frame3 = tk.Frame(row1, bg='#f0f0f0')
        frame3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame3, text="IFFT Image (Reconstructed)", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        label3 = tk.Label(frame3, bg='black', relief=tk.SUNKEN, bd=2)
        label3.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img3_resized = self.resize_for_display(ifft_img, max_width=350, max_height=350)
        image3 = Image.fromarray(img3_resized)
        photo3 = ImageTk.PhotoImage(image=image3)
        label3.config(image=photo3)
        label3.image = photo3
        
        # Hàng 2: Từng kênh R, G, B (nếu có)
        if channels_magnitude is not None:
            row2 = tk.Frame(scrollable_frame, bg='#f0f0f0')
            row2.pack(fill=tk.X, pady=5)
            
            # Kênh R
            frame_r = tk.Frame(row2, bg='#f0f0f0')
            frame_r.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(frame_r, text="Kênh R (Red) Spectrum", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='red').pack()
            label_r = tk.Label(frame_r, bg='black', relief=tk.SUNKEN, bd=2)
            label_r.pack(pady=5, fill=tk.BOTH, expand=True)
            
            img_r_resized = self.resize_for_display(channels_magnitude[0], max_width=350, max_height=350)
            image_r = Image.fromarray(img_r_resized)
            photo_r = ImageTk.PhotoImage(image=image_r)
            label_r.config(image=photo_r)
            label_r.image = photo_r
            
            # Kênh G
            frame_g = tk.Frame(row2, bg='#f0f0f0')
            frame_g.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(frame_g, text="Kênh G (Green) Spectrum", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='green').pack()
            label_g = tk.Label(frame_g, bg='black', relief=tk.SUNKEN, bd=2)
            label_g.pack(pady=5, fill=tk.BOTH, expand=True)
            
            img_g_resized = self.resize_for_display(channels_magnitude[1], max_width=350, max_height=350)
            image_g = Image.fromarray(img_g_resized)
            photo_g = ImageTk.PhotoImage(image=image_g)
            label_g.config(image=photo_g)
            label_g.image = photo_g
            
            # Kênh B
            frame_b = tk.Frame(row2, bg='#f0f0f0')
            frame_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(frame_b, text="Kênh B (Blue) Spectrum", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='blue').pack()
            label_b = tk.Label(frame_b, bg='black', relief=tk.SUNKEN, bd=2)
            label_b.pack(pady=5, fill=tk.BOTH, expand=True)
            
            img_b_resized = self.resize_for_display(channels_magnitude[2], max_width=350, max_height=350)
            image_b = Image.fromarray(img_b_resized)
            photo_b = ImageTk.PhotoImage(image=image_b)
            label_b.config(image=photo_b)
            label_b.image = photo_b
    
    def show_frequency_filter_results_window(self, original_img, filtered_img, filtered_channels):
        """Hiển thị kết quả lọc trong cửa sổ mới: Ảnh gốc, từng kênh, và kết quả"""
        # Tạo cửa sổ mới
        filter_window = tk.Toplevel(self.window)
        filter_window.title("Lọc thông thấp - Kết quả")
        filter_window.geometry("1600x1000")
        filter_window.configure(bg='#f0f0f0')
        
        # Container chính với scroll
        canvas = tk.Canvas(filter_window, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(filter_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Hàng 1: Ảnh gốc và kết quả
        row1 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row1.pack(fill=tk.X, pady=5)
        
        # Ảnh gốc
        frame_original = tk.Frame(row1, bg='#f0f0f0')
        frame_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_original, text="Ảnh gốc", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        label_original = tk.Label(frame_original, bg='black', relief=tk.SUNKEN, bd=2)
        label_original.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_orig_resized = self.resize_for_display(original_img, max_width=350, max_height=350)
        image_orig = Image.fromarray(img_orig_resized)
        photo_orig = ImageTk.PhotoImage(image=image_orig)
        label_original.config(image=photo_orig)
        label_original.image = photo_orig
        
        # Ảnh sau lọc
        frame_result = tk.Frame(row1, bg='#f0f0f0')
        frame_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_result, text="Ảnh sau lọc", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        label_result = tk.Label(frame_result, bg='black', relief=tk.SUNKEN, bd=2)
        label_result.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_result_resized = self.resize_for_display(filtered_img, max_width=350, max_height=350)
        image_result = Image.fromarray(img_result_resized)
        photo_result = ImageTk.PhotoImage(image=image_result)
        label_result.config(image=photo_result)
        label_result.image = photo_result
        
        # Frame chứa button và 3 kênh (nếu có)
        if filtered_channels is not None:
            # Frame cho button
            button_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
            button_frame.pack(fill=tk.X, pady=10)
            
            # Tạo một frame để chứa 3 kênh (ban đầu ẩn)
            channels_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
            
            # Tạo các widget cho 3 kênh (nhưng chưa pack)
            row2 = tk.Frame(channels_frame, bg='#f0f0f0')
            
            # Kênh R
            frame_r = tk.Frame(row2, bg='#f0f0f0')
            frame_r.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(frame_r, text="Kênh R (Red)", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='red').pack()
            label_r = tk.Label(frame_r, bg='black', relief=tk.SUNKEN, bd=2)
            label_r.pack(pady=5, fill=tk.BOTH, expand=True)
            
            # Tạo ảnh RGB từ kênh R (để hiển thị màu đỏ)
            R_channel = np.zeros_like(filtered_img)
            R_channel[:, :, 0] = filtered_channels[0]
            img_r_resized = self.resize_for_display(R_channel, max_width=350, max_height=350)
            image_r = Image.fromarray(img_r_resized)
            photo_r = ImageTk.PhotoImage(image=image_r)
            label_r.config(image=photo_r)
            label_r.image = photo_r
            
            # Kênh G
            frame_g = tk.Frame(row2, bg='#f0f0f0')
            frame_g.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(frame_g, text="Kênh G (Green)", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='green').pack()
            label_g = tk.Label(frame_g, bg='black', relief=tk.SUNKEN, bd=2)
            label_g.pack(pady=5, fill=tk.BOTH, expand=True)
            
            # Tạo ảnh RGB từ kênh G (để hiển thị màu xanh lá)
            G_channel = np.zeros_like(filtered_img)
            G_channel[:, :, 1] = filtered_channels[1]
            img_g_resized = self.resize_for_display(G_channel, max_width=350, max_height=350)
            image_g = Image.fromarray(img_g_resized)
            photo_g = ImageTk.PhotoImage(image=image_g)
            label_g.config(image=photo_g)
            label_g.image = photo_g
            
            # Kênh B
            frame_b = tk.Frame(row2, bg='#f0f0f0')
            frame_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(frame_b, text="Kênh B (Blue)", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='blue').pack()
            label_b = tk.Label(frame_b, bg='black', relief=tk.SUNKEN, bd=2)
            label_b.pack(pady=5, fill=tk.BOTH, expand=True)
            
            # Tạo ảnh RGB từ kênh B (để hiển thị màu xanh dương)
            B_channel = np.zeros_like(filtered_img)
            B_channel[:, :, 2] = filtered_channels[2]
            img_b_resized = self.resize_for_display(B_channel, max_width=350, max_height=350)
            image_b = Image.fromarray(img_b_resized)
            photo_b = ImageTk.PhotoImage(image=image_b)
            label_b.config(image=photo_b)
            label_b.image = photo_b
            
            row2.pack(fill=tk.X, pady=5)
            
            # Biến để theo dõi trạng thái hiển thị
            channels_visible = {'value': False}
            
            def toggle_channels():
                """Hiển thị/ẩn 3 kênh"""
                if not channels_visible['value']:
                    # Hiển thị 3 kênh
                    channels_frame.pack(fill=tk.X, pady=5)
                    show_channels_button.config(text="Ẩn 3 kênh ảnh")
                    channels_visible['value'] = True
                else:
                    # Ẩn 3 kênh
                    channels_frame.pack_forget()
                    show_channels_button.config(text="Hiển thị 3 kênh ảnh")
                    channels_visible['value'] = False
            
            # Button để hiển thị/ẩn 3 kênh
            show_channels_button = tk.Button(
                button_frame, 
                text="Hiển thị 3 kênh ảnh",
                command=toggle_channels,
                bg='#4CAF50',
                fg='white',
                font=('Arial', 11, 'bold'),
                width=20,
                height=2
            )
            show_channels_button.pack(pady=5)
    
    def resize_for_display(self, img, max_width=550, max_height=280):
        """Resize ảnh để hiển thị, giảm max_height để đảm bảo cả 2 ảnh đều hiển thị đầy đủ"""
        if img is None:
            return None
        
        # Đảm bảo ảnh là RGB format (3 channels)
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img.copy()
        
        # Đảm bảo ảnh là uint8
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        
        h, w = img_rgb.shape[:2]
        scale = min(max_width/w, max_height/h)
        if scale < 1:
            new_w, new_h = int(w*scale), int(h*scale)
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_rgb
    
    #Phuong phap su ly anh
    def neg_img(self, img):
        return 255 - img
    
    def log_transform(self, img, c):
        img_float = np.float32(img)
        log_img = c * np.log(1 + img_float)
        log_img = np.clip(log_img, 0, 255)
        return np.uint8(log_img)
    
    def piecewise_linear(self, img, low, high):
        output = np.zeros_like(img)
        output = np.where(img < low, img * (low / max(low, 1)), output)
        output = np.where((img >= low) & (img <= high), 
                         low + (img - low) * (255 - low) / max(high - low, 1), output)
        output = np.where(img > high, 255, output)
        return np.uint8(output)
    
    def gamma_transform(self, img, c, gamma):
        img_normalized = img / 255.0
        gamma_corrected = c * np.power(img_normalized, gamma)
        gamma_corrected = np.clip(gamma_corrected * 255, 0, 255)
        return np.uint8(gamma_corrected)

    @staticmethod
    def convolution_2d(A, k, b=0):
        kernel_high, kernel_width = k.shape
        original_h, original_w = A.shape
        
        if b > 0:
            pad_h = kernel_high // 2
            pad_w = kernel_width // 2
            A = np.pad(A, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        A_high, A_width = A.shape
        output_h = A_high - kernel_high + 1
        output_w = A_width - kernel_width + 1
        C = np.zeros((output_h, output_w))
        
        for i in range(output_h):
            for j in range(output_w):
                shape_A = A[i:i + kernel_high, j:j + kernel_width]
                C[i, j] = np.sum(shape_A * k)
        
        return C
    
    def convolution_rgb(self, img, kernel, b=0):
        if len(img.shape) == 2:
            return self.convolution_2d(img, kernel, b)
        else:
            channels = []
            for channel in range(3):
                conv_result = self.convolution_2d(img[:, :, channel], kernel, b)
                channels.append(conv_result)
            result = np.stack(channels, axis=2)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def average_filter(self, img, kernel_size):
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        return self.convolution_rgb(img, kernel, b=1)
    
    @staticmethod
    def Gausskernel(l, sigma):
        s = round((l-1)/2)
        ax = np.linspace(-s, s, l)
        gauss = np.exp(-np.square(ax) / (2 * np.square(sigma)))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)

    def gaussian_filter(self, img, kernel_size, sigma):
        kernel = self.Gausskernel(kernel_size, sigma)
        return self.convolution_rgb(img, kernel, b=1)
    
    def median_filter(self, img, kernel_size):
        pad = kernel_size // 2
        if len(img.shape) == 2:
            h, w = img.shape
            result = np.zeros_like(img)
            padded = np.pad(img, pad, mode='edge')
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    result[i, j] = np.median(window)
            return result
        else:
            result = np.zeros_like(img)
            for channel in range(3):
                h, w = img.shape[:2]
                padded = np.pad(img[:, :, channel], pad, mode='edge')
                for i in range(h):
                    for j in range(w):
                        window = padded[i:i+kernel_size, j:j+kernel_size]
                        result[i, j, channel] = np.median(window)
            return result.astype(np.uint8)
    
    def min_filter(self, img, kernel_size):
        pad = kernel_size // 2
        if len(img.shape) == 2:
            h, w = img.shape
            result = np.zeros_like(img)
            padded = np.pad(img, pad, mode='edge')
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    result[i, j] = np.min(window)
            return result
        else:
            result = np.zeros_like(img)
            for channel in range(3):
                h, w = img.shape[:2]
                padded = np.pad(img[:, :, channel], pad, mode='edge')
                for i in range(h):
                    for j in range(w):
                        window = padded[i:i+kernel_size, j:j+kernel_size]
                        result[i, j, channel] = np.min(window)
            return result.astype(np.uint8)
    
    def max_filter(self, img, kernel_size):
        pad = kernel_size // 2
        if len(img.shape) == 2:
            h, w = img.shape
            result = np.zeros_like(img)
            padded = np.pad(img, pad, mode='edge')
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    result[i, j] = np.max(window)
            return result
        else:
            result = np.zeros_like(img)
            for channel in range(3):
                h, w = img.shape[:2]
                padded = np.pad(img[:, :, channel], pad, mode='edge')
                for i in range(h):
                    for j in range(w):
                        window = padded[i:i+kernel_size, j:j+kernel_size]
                        result[i, j, channel] = np.max(window)
            return result.astype(np.uint8)
    
    def midpoint_filter(self, img, kernel_size):
        pad = kernel_size // 2
        if len(img.shape) == 2:
            h, w = img.shape
            result = np.zeros_like(img, dtype=np.float32)
            padded = np.pad(img, pad, mode='edge')
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    min_val = float(np.min(window))
                    max_val = float(np.max(window))
                    result[i, j] = (min_val + max_val) / 2
            return result.astype(np.uint8)
        else:
            result = np.zeros_like(img, dtype=np.float32)
            for channel in range(3):
                h, w = img.shape[:2]
                padded = np.pad(img[:, :, channel], pad, mode='edge')
                for i in range(h):
                    for j in range(w):
                        window = padded[i:i+kernel_size, j:j+kernel_size]
                        min_val = float(np.min(window))
                        max_val = float(np.max(window))
                        result[i, j, channel] = (min_val + max_val) / 2
            return result.astype(np.uint8)


    def histogram_equalization_channel(self, channel):
        hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        equalized = np.interp(channel.flatten(), range(256), cdf_normalized)
        return equalized.reshape(channel.shape).astype(np.uint8)
    
    def histogram_equalization(self, img):
        if len(img.shape) == 2:
            return self.histogram_equalization_channel(img)
        else:
            img_float = img.astype(np.float32) / 255.0
            r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
            
            max_val = np.maximum(np.maximum(r, g), b)
            min_val = np.minimum(np.minimum(r, g), b)
            diff = max_val - min_val
            
            v = max_val
            s = np.where(max_val != 0, diff / max_val, 0)
            
            h = np.zeros_like(max_val)
            mask_r = (max_val == r) & (diff != 0)
            mask_g = (max_val == g) & (diff != 0)
            mask_b = (max_val == b) & (diff != 0)
            
            h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
            h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
            h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
            
            v_uint8 = (v * 255).astype(np.uint8)
            v_equalized = self.histogram_equalization_channel(v_uint8).astype(np.float32) / 255.0
            
            c = v_equalized * s
            x = c * (1 - np.abs(((h / 60) % 2) - 1))
            m = v_equalized - c
            
            result = np.zeros_like(img_float)
            
            mask = (h >= 0) & (h < 60)
            result[mask] = np.stack([c[mask], x[mask], np.zeros_like(c[mask])], axis=1)
            
            mask = (h >= 60) & (h < 120)
            result[mask] = np.stack([x[mask], c[mask], np.zeros_like(c[mask])], axis=1)
            
            mask = (h >= 120) & (h < 180)
            result[mask] = np.stack([np.zeros_like(c[mask]), c[mask], x[mask]], axis=1)
            
            mask = (h >= 180) & (h < 240)
            result[mask] = np.stack([np.zeros_like(c[mask]), x[mask], c[mask]], axis=1)
            
            mask = (h >= 240) & (h < 300)
            result[mask] = np.stack([x[mask], np.zeros_like(c[mask]), c[mask]], axis=1)
            
            mask = (h >= 300) & (h < 360)
            result[mask] = np.stack([c[mask], np.zeros_like(c[mask]), x[mask]], axis=1)
            
            result = (result + m[:, :, np.newaxis]) * 255
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def spatial_to_frequency_domain(self, img, measure_time=False):
        """
        Chuyển ảnh từ miền không gian sang miền tần số.
        Hỗ trợ cả ảnh màu và grayscale.
        """
        time_info = {}
        is_color = len(img.shape) == 3
        
        if is_color:
            #Tach 3 kenh mau
            start_time = time.time()
            R_xy = img[:, :, 0].astype(np.float32)
            G_xy = img[:, :, 1].astype(np.float32)
            B_xy = img[:, :, 2].astype(np.float32)
            time_info['Bước 1: Tách ảnh thành 3 kênh R,G,B'] = (time.time() - start_time) * 1000  # ms
            
            # Bien doi Fourier 2D tung kenh
            start_time = time.time()
            F_R = np.fft.fft2(R_xy)
            F_G = np.fft.fft2(G_xy)
            F_B = np.fft.fft2(B_xy)
            time_info['Bước 2: FFT 2D từng kênh → FR, FG, FB'] = (time.time() - start_time) * 1000  # ms
            
            # Shift tâm về giữa
            start_time = time.time()
            F_R_shifted = np.fft.fftshift(F_R)
            F_G_shifted = np.fft.fftshift(F_G)
            F_B_shifted = np.fft.fftshift(F_B)
            time_info['Bước 3: Shift tâm về giữa'] = (time.time() - start_time) * 1000  # ms
            
            # Lay bien do va log tung kenh
            start_time = time.time()
            magnitude_R = 20 * np.log(np.abs(F_R_shifted) + 1e-10)
            magnitude_G = 20 * np.log(np.abs(F_G_shifted) + 1e-10)
            magnitude_B = 20 * np.log(np.abs(F_B_shifted) + 1e-10)
            time_info['Bước 4: Tính magnitude từng kênh'] = (time.time() - start_time) * 1000  # ms
            
            # Chuan hoa ve [0, 255] cho tung kenh
            start_time = time.time()
            magnitude_all = np.stack([magnitude_R, magnitude_G, magnitude_B], axis=2)
            magnitude_max = np.max(magnitude_all)
            magnitude_min = np.min(magnitude_all)
            if magnitude_max > magnitude_min:
                magnitude_normalized = ((magnitude_all - magnitude_min) / (magnitude_max - magnitude_min)) * 255.0
            else:
                magnitude_normalized = np.zeros_like(magnitude_all)
            time_info['Bước 5: Chuẩn hóa về [0,255]'] = (time.time() - start_time) * 1000  # ms
            
            result = magnitude_normalized.astype(np.uint8)
            F_uv_shifted = np.stack([F_R_shifted, F_G_shifted, F_B_shifted], axis=2)
            
            # Chuẩn hóa tung kenh magnitude de hien thi rieng
            magnitude_R_norm = ((magnitude_R - magnitude_min) / (magnitude_max - magnitude_min)) * 255.0 if magnitude_max > magnitude_min else np.zeros_like(magnitude_R)
            magnitude_G_norm = ((magnitude_G - magnitude_min) / (magnitude_max - magnitude_min)) * 255.0 if magnitude_max > magnitude_min else np.zeros_like(magnitude_G)
            magnitude_B_norm = ((magnitude_B - magnitude_min) / (magnitude_max - magnitude_min)) * 255.0 if magnitude_max > magnitude_min else np.zeros_like(magnitude_B)
            
            # Tao anh RGB tu tung kenh de hien thi rieng
            channel_R = np.stack([magnitude_R_norm.astype(np.uint8), 
                                 np.zeros_like(magnitude_R_norm, dtype=np.uint8), 
                                 np.zeros_like(magnitude_R_norm, dtype=np.uint8)], axis=2)
            channel_G = np.stack([np.zeros_like(magnitude_G_norm, dtype=np.uint8),
                                 magnitude_G_norm.astype(np.uint8),
                                 np.zeros_like(magnitude_G_norm, dtype=np.uint8)], axis=2)
            channel_B = np.stack([np.zeros_like(magnitude_B_norm, dtype=np.uint8),
                                 np.zeros_like(magnitude_B_norm, dtype=np.uint8),
                                 magnitude_B_norm.astype(np.uint8)], axis=2)
            
            channels_magnitude = [channel_R, channel_G, channel_B]
        else:
            # Xử lý ảnh grayscale
            # Bước 1: Lấy ảnh gốc f(x,y)
            start_time = time.time()
            f_xy = img.copy().astype(np.float32)
            time_info['Bước 1: Lấy ảnh gốc f(x,y)'] = (time.time() - start_time) * 1000  # ms
            
            # Bước 2: F(u,v) = F{f(x,y)}
            start_time = time.time()
            F_uv = np.fft.fft2(f_xy)
            time_info['Bước 2: FFT 2D → F(u,v)'] = (time.time() - start_time) * 1000  # ms
            
            # Bước 3: Shift tâm về giữa 
            start_time = time.time()
            F_uv_shifted = np.fft.fftshift(F_uv)
            time_info['Bước 3: Shift tâm về giữa'] = (time.time() - start_time) * 1000  # ms
            
            # Bước 4 & 5: Lấy biên độ và log
            start_time = time.time()
            magnitude_spectrum = 20 * np.log(np.abs(F_uv_shifted) + 1e-10)
            time_info['Bước 4: Tính magnitude'] = (time.time() - start_time) * 1000  # ms
            
            # Chuẩn hóa về [0, 255] 
            start_time = time.time()
            magnitude_max = np.max(magnitude_spectrum)
            magnitude_min = np.min(magnitude_spectrum)
            if magnitude_max > magnitude_min:
                magnitude_normalized = ((magnitude_spectrum - magnitude_min) / (magnitude_max - magnitude_min)) * 255.0
            else:
                magnitude_normalized = np.zeros_like(magnitude_spectrum)
            time_info['Bước 5: Chuẩn hóa về [0,255]'] = (time.time() - start_time) * 1000  # ms
            
            result = magnitude_normalized.astype(np.uint8)
            channels_magnitude = None
        
        if measure_time:
            if is_color:
                return result, F_uv_shifted, time_info, channels_magnitude
            else:
                return result, F_uv_shifted, time_info, None
        else:
            if is_color:
                return result, F_uv_shifted, channels_magnitude
            else:
                return result, F_uv_shifted, None
    
    def frequency_to_spatial_domain(self, F_uv_shifted, measure_time=False):
        """
        Chuyển ngược từ miền tần số về miền không gian.
        Hỗ trợ cả ảnh màu và grayscale.
        """
        time_info = {}
        is_color = len(F_uv_shifted.shape) == 3
        
        if is_color:
            # Xử lý ảnh màu: xử lý từng kênh riêng biệt
            # Tách thành 3 kênh FR(u,v), FG(u,v), FB(u,v)
            F_R_shifted = F_uv_shifted[:, :, 0]
            F_G_shifted = F_uv_shifted[:, :, 1]
            F_B_shifted = F_uv_shifted[:, :, 2]
            
            # Chuyển ngược tâm về góc (ifftshift)
            start_time = time.time()
            F_R_ishift = np.fft.ifftshift(F_R_shifted)
            F_G_ishift = np.fft.ifftshift(F_G_shifted)
            F_B_ishift = np.fft.ifftshift(F_B_shifted)
            time_info['IFFT Shift (chuyển ngược tâm)'] = (time.time() - start_time) * 1000  # ms
            
            # Inverse Fourier cho từng kênh
            # r(x,y) = F^-1{GR}, g(x,y) = F^-1{GG}, b(x,y) = F^-1{GB}
            start_time = time.time()
            r_xy = np.fft.ifft2(F_R_ishift)
            g_xy = np.fft.ifft2(F_G_ishift)
            b_xy = np.fft.ifft2(F_B_ishift)
            time_info['IFFT 2D từng kênh'] = (time.time() - start_time) * 1000  # ms
            
            # Lấy phần thực và chuẩn hóa
            start_time = time.time()
            r_real = np.real(r_xy)
            g_real = np.real(g_xy)
            b_real = np.real(b_xy)
            r_normalized = np.clip(r_real, 0, 255).astype(np.uint8)
            g_normalized = np.clip(g_real, 0, 255).astype(np.uint8)
            b_normalized = np.clip(b_real, 0, 255).astype(np.uint8)
            time_info['Chuẩn hóa IFFT'] = (time.time() - start_time) * 1000  # ms
            
            # Ghép lại 3 kênh: r(x,y), g(x,y), b(x,y) → RGB image
            f_xy_rgb = np.stack([r_normalized, g_normalized, b_normalized], axis=2)
        else:
            # Xử lý ảnh grayscale
            # Chuyển ngược tâm về góc (ifftshift)
            start_time = time.time()
            F_ishift = np.fft.ifftshift(F_uv_shifted)
            time_info['IFFT Shift (chuyển ngược tâm)'] = (time.time() - start_time) * 1000  # ms
            
            #  f(x,y) = F^{-1}{F(u,v)}
            start_time = time.time()
            f_xy_reconstructed = np.fft.ifft2(F_ishift)
            time_info['IFFT 2D'] = (time.time() - start_time) * 1000  # ms
            
            # Lấy phần thực và chuẩn hóa
            start_time = time.time()
            f_xy_real = np.real(f_xy_reconstructed)
            f_xy_normalized = np.clip(f_xy_real, 0, 255).astype(np.uint8)
            time_info['Chuẩn hóa IFFT'] = (time.time() - start_time) * 1000  # ms
            
            # Chuyển sang RGB để hiển thị
            f_xy_rgb = cv2.cvtColor(f_xy_normalized, cv2.COLOR_GRAY2RGB)
        
        if measure_time:
            return f_xy_rgb, time_info
        return f_xy_rgb
    
    def create_gaussian_lowpass_filter(self, M, N, D0):
        """
        Gaussian Low-pass Filter trong miền tần số.
        
        H(u,v) = e^(-D(u,v)^2 / (2*D0^2))
        
        Tham số:
            M, N: Kích thước ảnh (chiều cao, chiều rộng)
            D0: Cutoff frequency
        """
        # u: từ 0 đến N-1 (chiều ngang)
        # v: từ 0 đến M-1 (chiều dọc)
        u = np.arange(N)
        v = np.arange(M)
        u, v = np.meshgrid(u, v)
        
        # Tính khoảng cách từ tâm (M/2, N/2)
        # D(u,v) = sqrt((u - N/2)^2 + (v - M/2)^2)
        center_u = N / 2
        center_v = M / 2
        D = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        
        # Gaussian Low-pass Filter
        H = np.exp(-(D**2) / (2 * D0**2))
        
        return H
    
    def create_gaussian_highpass_filter(self, M, N, D0):
        """
        Gaussian High-pass Filter trong miền tần số.
        
        H_HP(u,v) = 1 - H_LP(u,v) = 1 - e^(-D(u,v)^2 / (2*D0^2))
        
        Tham số:
            M, N: Kích thước ảnh (chiều cao, chiều rộng)
            D0: Cutoff frequency
        """
        # u: từ 0 đến N-1 (chiều ngang)
        # v: từ 0 đến M-1 (chiều dọc)
        u = np.arange(N)
        v = np.arange(M)
        u, v = np.meshgrid(u, v)
        
        # Tính khoảng cách từ tâm (M/2, N/2)
        # D(u,v) = sqrt((u - N/2)^2 + (v - M/2)^2)
        center_u = N / 2
        center_v = M / 2
        D = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        
        # Gaussian High-pass Filter
        # H_HP(u,v) = 1 - e^(-D(u,v)^2 / (2*D0^2))
        H_LP = np.exp(-(D**2) / (2 * D0**2))
        H = 1 - H_LP
        
        return H
    
    def create_butterworth_lowpass_filter(self, M, N, D0, n=2):
        """
        Butterworth Low-pass Filter trong miền tần số.
        
        H(u,v) = 1 / (1 + (D(u,v)/D0)^(2*n))
        
        Tham số:
            M, N: Kích thước ảnh (chiều cao, chiều rộng)
            D0: Cutoff frequency
            n: Bậc của filter
        """
        # Tạo meshgrid với tâm ở giữa
        # u: từ 0 đến N-1 (chiều ngang)
        # v: từ 0 đến M-1 (chiều dọc)
        u = np.arange(N)
        v = np.arange(M)
        u, v = np.meshgrid(u, v)
        
        # Tính khoảng cách từ tâm (M/2, N/2)
        # D(u,v) = sqrt((u - N/2)^2 + (v - M/2)^2)
        center_u = N / 2
        center_v = M / 2
        D = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        
        # Butterworth Low-pass Filter
        H = 1 / (1 + (D / D0)**(2 * n))
        
        return H
    
    def create_butterworth_highpass_filter(self, M, N, D0, n=2):
        """
        Butterworth High-pass Filter trong miền tần số.
        
        H(u,v) = 1 / (1 + (D0/D(u,v))^(2*n))
        
        Tham số:
            M, N: Kích thước ảnh (chiều cao, chiều rộng)
            D0: Cutoff frequency
            n: Bậc của filter
        """
        # Tạo meshgrid với tâm ở giữa
        # u: từ 0 đến N-1 (chiều ngang)
        # v: từ 0 đến M-1 (chiều dọc)
        u = np.arange(N)
        v = np.arange(M)
        u, v = np.meshgrid(u, v)
        
        # Tính khoảng cách từ tâm (M/2, N/2)
        # D(u,v) = sqrt((u - N/2)^2 + (v - M/2)^2)
        center_u = N / 2
        center_v = M / 2
        D = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        
        # Tránh chia cho 0: thay thế D = 0 bằng giá trị rất nhỏ
        D = np.where(D == 0, 1e-10, D)
        
        # Butterworth High-pass Filter
        H = 1 / (1 + (D0 / D)**(2 * n))
        
        return H
    
    def create_ideal_lowpass_filter(self, M, N, D0):
        """
        Ideal Low-pass Filter trong miền tần số.
        
        H(u,v) = {
            1, nếu D(u,v) ≤ D0
            0, nếu D(u,v) > D0
        }
        
        Trong đó: D(u,v) = sqrt((u - N/2)^2 + (v - M/2)^2)
        
        Tham số:
            M, N: Kích thước ảnh (chiều cao, chiều rộng)
            D0: Cutoff frequency (bán kính)
        """
        # Tạo meshgrid với tâm ở giữa
        # u: từ 0 đến N-1 (chiều ngang)
        # v: từ 0 đến M-1 (chiều dọc)
        u = np.arange(N)
        v = np.arange(M)
        u, v = np.meshgrid(u, v)
        
        # Tính khoảng cách từ tâm (M/2, N/2)
        # D(u,v) = sqrt((u - N/2)^2 + (v - M/2)^2)
        center_u = N / 2
        center_v = M / 2
        D = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        
        # Ideal Low-pass Filter
        # H(u,v) = 1 nếu D(u,v) ≤ D0, ngược lại = 0
        H = np.where(D <= D0, 1.0, 0.0)
        
        return H
    
    def create_ideal_highpass_filter(self, M, N, D0):
        """
        Ideal High-pass Filter trong miền tần số.
        
        H(u,v) = {
            0, nếu D(u,v) ≤ D0
            1, nếu D(u,v) > D0
        }
        
        Trong đó: D(u,v) = sqrt((u - N/2)^2 + (v - M/2)^2)
        
        Tham số:
            M, N: Kích thước ảnh (chiều cao, chiều rộng)
            D0: Cutoff frequency (bán kính)
        """
        # Tạo meshgrid với tâm ở giữa
        # u: từ 0 đến N-1 (chiều ngang)
        # v: từ 0 đến M-1 (chiều dọc)
        u = np.arange(N)
        v = np.arange(M)
        u, v = np.meshgrid(u, v)
        
        # Tính khoảng cách từ tâm (M/2, N/2)
        # D(u,v) = sqrt((u - N/2)^2 + (v - M/2)^2)
        center_u = N / 2
        center_v = M / 2
        D = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        
        # Ideal High-pass Filter
        # H(u,v) = 0 nếu D(u,v) ≤ D0, ngược lại = 1
        H = np.where(D <= D0, 0.0, 1.0)
        
        return H
    
    def apply_frequency_lowpass_filter(self, img, filter_type="Gaussian", D0=40, n=2):
        """
        Áp dụng lọc trong miền tần số cho ảnh màu và grayscale.
        Sử dụng các hàm Fourier đã được xây dựng sẵn.
        
        Các bước:
        1. Chuyển ảnh sang miền tần số: F(u,v) = F{f(x,y)} (sử dụng spatial_to_frequency_domain)
        2. Áp dụng bộ lọc H(u,v): G(u,v) = F(u,v) * H(u,v)
        3. Chuyển ngược về miền không gian: g(x,y) = F^-1{G(u,v)} (sử dụng frequency_to_spatial_domain)
        """
        # Bước 1: Chuyển sang miền tần số (sử dụng hàm đã có)
        _, F_uv_shifted, _ = self.spatial_to_frequency_domain(img, measure_time=False)
        
        # Lấy kích thước ảnh để tạo filter
        if len(img.shape) == 3:
            M, N = img.shape[:2]
        else:
            M, N = img.shape
        
        # Tạo filter H(u,v)
        if filter_type == "Gaussian":
            H_uv = self.create_gaussian_lowpass_filter(M, N, D0)
        elif filter_type == "Gaussian High-pass":
            H_uv = self.create_gaussian_highpass_filter(M, N, D0)
        elif filter_type == "Butterworth":
            H_uv = self.create_butterworth_lowpass_filter(M, N, D0, n)
        elif filter_type == "Butterworth High-pass":
            H_uv = self.create_butterworth_highpass_filter(M, N, D0, n)
        elif filter_type == "Ideal":
            H_uv = self.create_ideal_lowpass_filter(M, N, D0)
        else:  # Ideal High-pass
            H_uv = self.create_ideal_highpass_filter(M, N, D0)
        
        # Bước 2: Áp dụng filter: G(u,v) = F(u,v) * H(u,v)
        if len(F_uv_shifted.shape) == 3:
            # Ảnh màu: áp dụng filter cho từng kênh
            G_uv_shifted = F_uv_shifted * H_uv[:, :, np.newaxis]
        else:
            # Ảnh grayscale
            G_uv_shifted = F_uv_shifted * H_uv
        
        # Bước 3: Chuyển ngược về miền không gian (sử dụng hàm đã có)
        filtered_img_rgb = self.frequency_to_spatial_domain(G_uv_shifted, measure_time=False)
        
        # Xử lý kết quả
        if len(img.shape) == 3:
            # Ảnh màu: trả về filtered_img và filtered_channels
            filtered_img = filtered_img_rgb
            # Tách các kênh để hiển thị riêng
            filtered_channels = [
                filtered_img[:, :, 0],
                filtered_img[:, :, 1],
                filtered_img[:, :, 2]
            ]
            return filtered_img, filtered_channels
        else:
            # Ảnh grayscale: chuyển từ RGB về grayscale
            filtered_img = cv2.cvtColor(filtered_img_rgb, cv2.COLOR_RGB2GRAY)
            return filtered_img, None
    
    def apply_frequency_filter(self):
        """Áp dụng lọc thông thấp trong miền tần số và hiển thị kết quả"""
        if self.original_image is None:
            return
        
        # Lấy tham số từ UI
        filter_type_value = self.filter_type.get()
        
        # Lấy D0 từ slider tương ứng với filter được chọn
        if filter_type_value == "gaussian":
            filter_type = "Gaussian"
            D0 = self.filter_d0_gaussian.get()
            n = 2  # Không dùng cho Gaussian
        elif filter_type_value == "gaussian_highpass":
            filter_type = "Gaussian High-pass"
            D0 = self.filter_d0_gaussian_highpass.get()
            n = 2  # Không dùng cho Gaussian High-pass
        elif filter_type_value == "butterworth":
            filter_type = "Butterworth"
            D0 = self.filter_d0_butterworth.get()
            n = int(self.filter_n.get())
        elif filter_type_value == "butterworth_highpass":
            filter_type = "Butterworth High-pass"
            D0 = self.filter_d0_butterworth_highpass.get()
            n = int(self.filter_n_butterworth_highpass.get())
        elif filter_type_value == "ideal":
            filter_type = "Ideal"
            D0 = self.filter_d0_ideal.get()
            n = 2  # Không dùng cho Ideal
        else:  # ideal_highpass
            filter_type = "Ideal High-pass"
            D0 = self.filter_d0_ideal_highpass.get()
            n = 2  # Không dùng cho Ideal High-pass
        
        # Áp dụng filter
        result = self.apply_frequency_lowpass_filter(
            self.original_image, 
            filter_type=filter_type, 
            D0=D0, 
            n=n
        )
        
        # Kết quả trả về có thể là (filtered_img, None) hoặc (filtered_img, filtered_channels)
        if isinstance(result, tuple):
            filtered_img, filtered_channels = result
        else:
            filtered_img = result
            filtered_channels = None
        
        # Lưu filtered_channels để có thể hiển thị sau khi nhấn nút
        self.last_filtered_channels = filtered_channels
        self.last_filtered_img = filtered_img
        
        # Cập nhật trạng thái nút hiển thị 3 kênh
        if hasattr(self, 'show_channels_button'):
            if filtered_channels is not None:
                self.show_channels_button.config(state=tk.NORMAL)
            else:
                self.show_channels_button.config(state=tk.DISABLED)
        
        # Hiển thị kết quả trên UI chính (không tự động mở cửa sổ mới)
        self.processed_image = filtered_img
        filter_name = f"Lọc thông thấp ({filter_type}, D0={D0})"
        if filter_type == "Butterworth" or filter_type == "Butterworth High-pass":
            filter_name += f", n={n}"
        self.processed_title_label.config(text=filter_name)
        self.show_processed_image(self.processed_image)
    
    def show_channels_window(self):
        """Mở cửa sổ hiển thị 3 kênh ảnh khi người dùng nhấn nút"""
        if not hasattr(self, 'last_filtered_channels') or self.last_filtered_channels is None:
            return
        
        if not hasattr(self, 'original_image') or self.original_image is None:
            return
        
        # Mở cửa sổ hiển thị 3 kênh
        self.show_frequency_filter_results_window(
            self.original_image,
            self.last_filtered_img,
            self.last_filtered_channels
        )
    
    def compare_gaussian_filters(self):
        """So sánh Gaussian filter giữa miền không gian và miền tần số"""
        if self.original_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        
        # Lấy tham số từ UI
        kernel_size = int(self.compare_gauss_kernel.get())
        sigma = self.compare_gauss_sigma.get()
        D0 = self.compare_gauss_d0.get()
        
        # Đo thời gian cho Gaussian filter trong miền không gian
        spatial_start = time.time()
        spatial_result = self.gaussian_filter(self.original_image.copy(), kernel_size, sigma)
        spatial_time = (time.time() - spatial_start) * 1000  # ms
        
        # Đo thời gian cho Gaussian filter trong miền tần số
        freq_start = time.time()
        freq_result, _ = self.apply_frequency_lowpass_filter(
            self.original_image.copy(),
            filter_type="Gaussian",
            D0=D0,
            n=2
        )
        freq_time = (time.time() - freq_start) * 1000  # ms
        
        # In thời gian ra console
        print("\n=== SO SÁNH GAUSSIAN FILTER ===")
        print(f"Miền không gian: {spatial_time:.2f} ms")
        print(f"Miền tần số: {freq_time:.2f} ms")
        print(f"Tỷ lệ: {freq_time/spatial_time:.2f}x")
        print("=" * 30)
        
        # Hiển thị kết quả so sánh
        self.show_gaussian_comparison_window(
            self.original_image,
            spatial_result,
            freq_result,
            kernel_size,
            sigma,
            D0,
            spatial_time,
            freq_time
        )
    
    def show_gaussian_comparison_window(self, original_img, spatial_result, freq_result, kernel_size, sigma, D0, spatial_time, freq_time):
        """Hiển thị cửa sổ so sánh Gaussian filter giữa 2 miền"""
        # Tạo cửa sổ mới
        compare_window = tk.Toplevel(self.window)
        compare_window.title("So sánh Gaussian Filter")
        compare_window.geometry("1400x700")
        compare_window.configure(bg='#f0f0f0')
        
        # Container chính
        main_frame = tk.Frame(compare_window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tiêu đề
        title_label = tk.Label(main_frame, text="So sánh Gaussian Filter", 
                              font=('Arial', 14, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=(0, 5))
        
        # Frame hiển thị thông số thời gian
        time_frame = tk.Frame(main_frame, bg='#e3f2fd', relief=tk.RAISED, bd=2)
        time_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Thông số thời gian miền không gian
        spatial_time_frame = tk.Frame(time_frame, bg='#e3f2fd')
        spatial_time_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        tk.Label(spatial_time_frame, text="Miền không gian", 
                font=('Arial', 11, 'bold'), bg='#e3f2fd').pack()
        tk.Label(spatial_time_frame, text=f"Thời gian: {spatial_time:.2f} ms", 
                font=('Arial', 10), bg='#e3f2fd', fg='#1976d2').pack()
        
        # Thông số thời gian miền tần số
        freq_time_frame = tk.Frame(time_frame, bg='#e3f2fd')
        freq_time_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        tk.Label(freq_time_frame, text="Miền tần số", 
                font=('Arial', 11, 'bold'), bg='#e3f2fd').pack()
        tk.Label(freq_time_frame, text=f"Thời gian: {freq_time:.2f} ms", 
                font=('Arial', 10), bg='#e3f2fd', fg='#1976d2').pack()
        
        # Tỷ lệ so sánh
        ratio = freq_time / spatial_time if spatial_time > 0 else 0
        ratio_frame = tk.Frame(time_frame, bg='#e3f2fd')
        ratio_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        tk.Label(ratio_frame, text="Tỷ lệ", 
                font=('Arial', 11, 'bold'), bg='#e3f2fd').pack()
        ratio_color = '#d32f2f' if ratio > 1 else '#388e3c'
        tk.Label(ratio_frame, text=f"{ratio:.2f}x", 
                font=('Arial', 10, 'bold'), bg='#e3f2fd', fg=ratio_color).pack()
        
        # Hàng chứa 3 ảnh
        images_frame = tk.Frame(main_frame, bg='#f0f0f0')
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Ảnh gốc
        frame_original = tk.Frame(images_frame, bg='#f0f0f0')
        frame_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_original, text="Ảnh gốc", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        label_original = tk.Label(frame_original, bg='black', relief=tk.SUNKEN, bd=2)
        label_original.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_orig_resized = self.resize_for_display(original_img, max_width=400, max_height=400)
        image_orig = Image.fromarray(img_orig_resized)
        photo_orig = ImageTk.PhotoImage(image=image_orig)
        label_original.config(image=photo_orig)
        label_original.image = photo_orig
        
        # Ảnh sau lọc miền không gian
        frame_spatial = tk.Frame(images_frame, bg='#f0f0f0')
        frame_spatial.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        spatial_title = f"Miền không gian\n(kernel={kernel_size}, σ={sigma:.1f})\nThời gian: {spatial_time:.2f} ms"
        tk.Label(frame_spatial, text=spatial_title, font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        label_spatial = tk.Label(frame_spatial, bg='black', relief=tk.SUNKEN, bd=2)
        label_spatial.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_spatial_resized = self.resize_for_display(spatial_result, max_width=400, max_height=400)
        image_spatial = Image.fromarray(img_spatial_resized)
        photo_spatial = ImageTk.PhotoImage(image=image_spatial)
        label_spatial.config(image=photo_spatial)
        label_spatial.image = photo_spatial
        
        # Ảnh sau lọc miền tần số
        frame_freq = tk.Frame(images_frame, bg='#f0f0f0')
        frame_freq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        freq_title = f"Miền tần số\n(D0={D0})\nThời gian: {freq_time:.2f} ms"
        tk.Label(frame_freq, text=freq_title, font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        label_freq = tk.Label(frame_freq, bg='black', relief=tk.SUNKEN, bd=2)
        label_freq.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_freq_resized = self.resize_for_display(freq_result, max_width=400, max_height=400)
        image_freq = Image.fromarray(img_freq_resized)
        photo_freq = ImageTk.PhotoImage(image=image_freq)
        label_freq.config(image=photo_freq)
        label_freq.image = photo_freq
    
    def laplacian_filter(self, img):
        # Laplacian kernel
        kernel = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=np.float32)
        
        if len(img.shape) == 2:
            return self.convolution_2d(img.astype(np.float32), kernel, b=1)
        else:
            channels = []
            for channel in range(3):
                conv_result = self.convolution_2d(img[:, :, channel].astype(np.float32), kernel, b=1)
                channels.append(conv_result)
            result = np.stack(channels, axis=2)
            return result
    
    def sobel_gradient(self, img):
        # Sobel kernels
        #x(ngang)
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        #y(doc)
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
        
        if len(img.shape) == 2:
            gx = self.convolution_2d(img.astype(np.float32), sobel_x, b=1)
            gy = self.convolution_2d(img.astype(np.float32), sobel_y, b=1)
            magnitude = np.sqrt(gx**2 + gy**2)
            return magnitude
        else:
            channels = []
            for channel in range(3):
                gx = self.convolution_2d(img[:, :, channel].astype(np.float32), sobel_x, b=1)
                gy = self.convolution_2d(img[:, :, channel].astype(np.float32), sobel_y, b=1)
                magnitude = np.sqrt(gx**2 + gy**2)
                channels.append(magnitude)
            result = np.stack(channels, axis=2)
            return result
    
    def convert_to_binary(self, img, threshold=127):
       
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.copy()
        
        if img_gray.dtype != np.uint8:
            if img_gray.max() > 255 or img_gray.min() < 0:
                img_gray = np.clip(img_gray, 0, 255)
            img_gray = img_gray.astype(np.uint8)
        
        _, img_binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        
        return img_binary
    
    def analyze_binary_image(self, img):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.copy()
        
        if img_gray.dtype != np.uint8:
            img_gray = img_gray.astype(np.uint8)
        
        unique_values = np.unique(img_gray)
        if len(unique_values) > 2:
            img_binary = self.convert_to_binary(img_gray, threshold=127)
        else:
            img_binary = img_gray
        
        img_binary = self.opening(img_binary, kernel_size=3)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binary, connectivity=8)
        
        num_objects = num_labels - 1
        
        object_areas = []
        for i in range(1, num_labels):  
            area = stats[i, cv2.CC_STAT_AREA]
            object_areas.append(area)
        
        print("\n" + "="*50)
        print("PHÂN TÍCH ẢNH NHỊ PHÂN")
        print("="*50)
        print(f"Số lượng đối tượng: {num_objects}")
        if num_objects > 0:
            print(f"\nDiện tích từng đối tượng (pixels):")
            for i, area in enumerate(object_areas, 1):
                print(f"  Đối tượng {i}: {area} pixels")
        else:
            print("\nKhông tìm thấy đối tượng nào trong ảnh!")
        print("="*50 + "\n")
        
        return {
            'num_objects': num_objects,
            'object_areas': object_areas,
            'labels': labels,
            'stats': stats,
            'centroids': centroids
        }
    
    def delation(self, img, kernel_size=3, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if len(img.shape) == 2:
            return cv2.dilate(img, kernel, iterations=iterations)
        else:
            channels = []
            for channel in range(3):
                dilated = cv2.dilate(img[:, :, channel], kernel, iterations=iterations)
                channels.append(dilated)
            result = np.stack(channels, axis=2)
            return result

    def erosion(self, img, kernel_size=3, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if len(img.shape) == 2:
            return cv2.erode(img, kernel, iterations=iterations)
        else:
            channels = []
            for channel in range(3):
                eroded = cv2.erode(img[:, :, channel], kernel, iterations=iterations)
                channels.append(eroded)
            result = np.stack(channels, axis=2)
            return result
    
    def opening(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if len(img.shape) == 2:
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        else:
            channels = []
            for channel in range(3):
                opened = cv2.morphologyEx(img[:, :, channel], cv2.MORPH_OPEN, kernel)
                channels.append(opened)
            result = np.stack(channels, axis=2)
            return result

    def closing(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if len(img.shape) == 2:
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        else:
            channels = []
            for channel in range(3):
                closed = cv2.morphologyEx(img[:, :, channel], cv2.MORPH_CLOSE, kernel)
                channels.append(closed)
            result = np.stack(channels, axis=2)
            return result
    
    def apply_morphological_filtering(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Vui lòng chọn ảnh trước!")
            return
        
        kernel_size = int(self.morph_kernel.get())
        
        img_to_process = self.processed_image if self.processed_image is not None else self.original_image
        
        if len(img_to_process.shape) == 3:
            img_gray = cv2.cvtColor(img_to_process, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_to_process.copy()
        
        if img_gray.dtype != np.uint8:
            img_gray = img_gray.astype(np.uint8)
        
        unique_values = np.unique(img_gray)
        if len(unique_values) > 2:
            _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        else:
            img_binary = img_gray
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        #a
        img_noisy = img_binary.copy()
        
        # c Erosion
        img_eroded = self.erosion(img_noisy, kernel_size=kernel_size, iterations=1)
        
        # d Erosion rồi Dilation
        img_opened = self.opening(img_noisy, kernel_size=kernel_size)
        
        # (e) Ảnh được giãn nở từ kết quả phép mở
        img_dilated_from_opening = self.delation(img_opened, kernel_size=kernel_size, iterations=1)
        
        # (f) Closing của Opening
        img_closed_of_opening = self.closing(img_opened, kernel_size=kernel_size)
        
        self.show_morphological_filtering_results(
            img_noisy,           # (a)
            kernel,              # (b)
            img_eroded,          # (c)
            img_opened,          # (d)
            img_dilated_from_opening,  # (e)
            img_closed_of_opening,     
            kernel_size
        )
    
    def show_morphological_filtering_results(self, img_noisy, kernel, img_eroded, 
                                            img_opened, img_dilated, img_closed, kernel_size):
        """Hiển thị tất cả các bước của quy trình lọc hình thái học"""
        # Tạo cửa sổ mới với kích thước phù hợp
        morph_window = tk.Toplevel(self.window)
        morph_window.title("Lọc hình thái học - Quy trình đầy đủ")
        morph_window.geometry("1400x900")
        morph_window.configure(bg='#f0f0f0')
        
        # Container với scroll
        scroll_frame = tk.Frame(morph_window, bg='#f0f0f0')
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(scroll_frame, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        def update_scrollregion(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", update_scrollregion)
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        canvas.bind('<Configure>', configure_canvas_width)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Tiêu đề
        title_label = tk.Label(scrollable_frame, 
                              text="Ứng dụng phép mở và phép đóng trong lọc hình thái học",
                              font=('Arial', 14, 'bold'), bg='#795548', fg='white', pady=10)
        title_label.pack(fill=tk.X, pady=(0, 10))
        
        # Thông tin kernel
        info_frame = tk.Frame(scrollable_frame, bg='#e3f2fd', relief=tk.RAISED, bd=2)
        info_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        tk.Label(info_frame, text=f"Kích thước phần tử cấu trúc: {kernel_size}x{kernel_size}",
                font=('Arial', 11, 'bold'), bg='#e3f2fd').pack(pady=5)
        
        # Kích thước ảnh hiển thị (nhỏ hơn để tránh tràn)
        img_display_size = 300
        
        # Hàm helper để tạo frame ảnh
        def create_image_frame(parent, title, img_data):
            frame = tk.Frame(parent, bg='#f0f0f0')
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(frame, text=title, 
                    font=('Arial', 11, 'bold'), bg='#f0f0f0').pack()
            label = tk.Label(frame, bg='black', relief=tk.SUNKEN, bd=2)
            label.pack(pady=5)
            
            img_resized = self.resize_for_display(img_data, max_width=img_display_size, max_height=img_display_size)
            image = Image.fromarray(img_resized)
            photo = ImageTk.PhotoImage(image=image)
            label.config(image=photo)
            label.image = photo
            return frame
        
        # Hàng 1: (a) và (b)
        row1 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row1.pack(fill=tk.X, pady=5)
        
        # (a) Ảnh có nhiễu
        create_image_frame(row1, "(a) Ảnh có nhiễu", img_noisy)
        
        # (b) Phần tử cấu trúc (kernel)
        frame_b = tk.Frame(row1, bg='#f0f0f0')
        frame_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_b, text=f"(b) Phần tử cấu trúc ({kernel_size}x{kernel_size})", 
                font=('Arial', 11, 'bold'), bg='#f0f0f0').pack()
        label_b = tk.Label(frame_b, bg='white', relief=tk.SUNKEN, bd=2)
        label_b.pack(pady=5)
        
        # Tạo ảnh hiển thị kernel (phóng đại để dễ nhìn)
        kernel_display_size = 200
        kernel_img = np.zeros((kernel_display_size, kernel_display_size), dtype=np.uint8)
        cell_size = kernel_display_size // kernel_size
        for i in range(kernel_size):
            for j in range(kernel_size):
                if kernel[i, j] == 1:
                    y1, y2 = i * cell_size, (i + 1) * cell_size
                    x1, x2 = j * cell_size, (j + 1) * cell_size
                    kernel_img[y1:y2, x1:x2] = 255
        
        kernel_img_resized = self.resize_for_display(kernel_img, max_width=img_display_size, max_height=img_display_size)
        image_b = Image.fromarray(kernel_img_resized)
        photo_b = ImageTk.PhotoImage(image=image_b)
        label_b.config(image=photo_b)
        label_b.image = photo_b
        
        # Hàng 2: (c) và (d)
        row2 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row2.pack(fill=tk.X, pady=5)
        
        create_image_frame(row2, "(c) Ảnh sau khi ăn mòn (Erosion)", img_eroded)
        create_image_frame(row2, "(d) Phép mở (Opening)", img_opened)
        
        # Hàng 3: (e) và (f)
        row3 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row3.pack(fill=tk.X, pady=5)
        
        create_image_frame(row3, "(e) Giãn nở từ kết quả phép mở", img_dilated)
        create_image_frame(row3, "(f) Phép đóng của kết quả phép mở", img_closed)
        
        # Thêm padding ở cuối để đảm bảo scroll đầy đủ
        bottom_padding = tk.Frame(scrollable_frame, bg='#f0f0f0', height=20)
        bottom_padding.pack(fill=tk.X)
        
        # In thông tin ra console
        print("\n" + "="*60)
        print("QUY TRÌNH LỌC HÌNH THÁI HỌC")
        print("="*60)
        print(f"Kích thước phần tử cấu trúc: {kernel_size}x{kernel_size}")
        print("\nCác bước thực hiện:")
        print("  (a) Ảnh có nhiễu: Ảnh gốc")
        print("  (b) Phần tử cấu trúc: Kernel", kernel_size, "x", kernel_size)
        print("  (c) Erosion: A ⊖ B")
        print("  (d) Opening: (A ⊖ B) ⊕ B")
        print("  (e) Dilation của Opening: ((A ⊖ B) ⊕ B) ⊕ B")
        print("  (f) Closing của Opening: (((A ⊖ B) ⊕ B) ⊕ B) ⊖ B")
        print("="*60 + "\n")
            
    
    def box_filter_5x5(self, img):
        return self.average_filter(img, 5)
    
    def line_detection(self, img):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.copy()
        
        img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        
        threshval = 50
        n = 255
        retval, img_binary = cv2.threshold(img_blurred, threshval, n, cv2.THRESH_BINARY)
        
        img_a = img_binary.copy()
        
        ddepth = cv2.CV_16S
        kernel_size = 3
        img_lap = cv2.Laplacian(img_binary, ddepth, ksize=kernel_size)
        
        laplacian_min = np.min(img_lap)
        laplacian_max = np.max(img_lap)
        laplacian_range = max(abs(laplacian_min), abs(laplacian_max))
        
        if laplacian_range > 0:
            laplacian_normalized = img_lap.astype(np.float32) / laplacian_range
            img_b = (laplacian_normalized * 127.0 + 128.0).astype(np.uint8)
            img_b = np.clip(img_b, 0, 255)
        else:
            img_b = np.full_like(img_binary, 128, dtype=np.uint8)
        
        img_lap_abs = cv2.convertScaleAbs(img_lap)
        img_c = img_lap_abs
        
        img_lap_pos = (img_lap > 0).astype(np.uint8) * 255
        img_d = img_lap_pos
        
        return img_a, img_b, img_c, img_d
    
    def adaptive_thresholding(self, img, thresh_val=132, block_size=11, c=2):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.copy()
        
        img_blurred = cv2.medianBlur(img_gray, 5)
        
        img_a = img_blurred.copy()
        
        ret, th1 = cv2.threshold(img_blurred, thresh_val, 255, cv2.THRESH_BINARY)
        img_b = th1
        
        th2 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, block_size, c)
        img_c = th2
        
        th3 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, block_size, c)
        img_d = th3
        
        return img_a, img_b, img_c, img_d
    
    def optimum_thresholding(self, img, block_size=7, c=2):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.copy()
        
        # blur , kernel 5
        img_blurred = cv2.medianBlur(img_gray, 5)
        
        # anh sau blur
        img_a = img_blurred.copy()
        
        ret, th1 = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_b = th1
        otsu_threshold = ret  
        
        th2 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, block_size, c)
        img_c = th2
        
        th3 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, block_size, c)
        img_d = th3
        
        return img_a, img_b, img_c, img_d, otsu_threshold

    def apply_adaptive_thresholding(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Vui lòng chọn ảnh trước!")
            return
        
        thresh_val = self.adaptive_thresh_val.get()
        block_size = self.adaptive_block_size.get()
        c = self.adaptive_c.get()
        
        img_a, img_b, img_c, img_d = self.adaptive_thresholding(
            self.original_image, thresh_val, block_size, c
        )
        self.show_adaptive_thresholding_results_window(img_a, img_b, img_c, img_d, thresh_val, block_size, c)
    
    def show_adaptive_thresholding_results_window(self, img_a, img_b, img_c, img_d, thresh_val, block_size, c):
        adaptive_window = tk.Toplevel(self.window)
        adaptive_window.title("Adaptive Thresholding - Kết quả")
        adaptive_window.geometry("1600x900")
        adaptive_window.configure(bg='#f0f0f0')
        
        # Container với scroll
        canvas = tk.Canvas(adaptive_window, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(adaptive_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mousewheel để scroll
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        # Bind khi chuột vào/ra khỏi cửa sổ
        adaptive_window.bind("<Enter>", bind_mousewheel)
        adaptive_window.bind("<Leave>", unbind_mousewheel)
        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_label = tk.Label(scrollable_frame, text="Adaptive Thresholding", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=(10, 20))
        
        # Thông tin về tham số
        info_text = f"Tham số: Threshold Value={thresh_val}, Block Size={block_size}, C={c}"
        info_label = tk.Label(scrollable_frame, text=info_text, 
                             font=('Arial', 10, 'italic'), bg='#f0f0f0', fg='gray')
        info_label.pack(pady=(0, 10))
        
        # Hàng 1: (a) Original và (b) Thresholding
        row1 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row1.pack(fill=tk.X, pady=5)
        
        # Ảnh (a): Original Image
        frame_a = tk.Frame(row1, bg='#f0f0f0')
        frame_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_a, text="(a) Original Image", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_a, text="(sau Median Blur)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_a = tk.Label(frame_a, bg='black', relief=tk.SUNKEN, bd=2)
        label_a.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_a_resized = self.resize_for_display(img_a, max_width=400, max_height=400)
        image_a = Image.fromarray(img_a_resized)
        photo_a = ImageTk.PhotoImage(image=image_a)
        label_a.config(image=photo_a)
        label_a.image = photo_a
        
        # Ảnh (b): Thresholding
        frame_b = tk.Frame(row1, bg='#f0f0f0')
        frame_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_b, text=f"(b) Thresholding (v = {thresh_val})", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_b, text="(THRESH_BINARY)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_b = tk.Label(frame_b, bg='black', relief=tk.SUNKEN, bd=2)
        label_b.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_b_resized = self.resize_for_display(img_b, max_width=400, max_height=400)
        image_b = Image.fromarray(img_b_resized)
        photo_b = ImageTk.PhotoImage(image=image_b)
        label_b.config(image=photo_b)
        label_b.image = photo_b
        
        # Hàng 2: (c) Mean Thresholding và (d) Gaussian Thresholding
        row2 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row2.pack(fill=tk.X, pady=5)
        
        # Ảnh (c): Mean Thresholding
        frame_c = tk.Frame(row2, bg='#f0f0f0')
        frame_c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_c, text="(c) Mean Thresholding", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_c, text="(ADAPTIVE_THRESH_MEAN_C)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_c = tk.Label(frame_c, bg='black', relief=tk.SUNKEN, bd=2)
        label_c.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_c_resized = self.resize_for_display(img_c, max_width=400, max_height=400)
        image_c = Image.fromarray(img_c_resized)
        photo_c = ImageTk.PhotoImage(image=image_c)
        label_c.config(image=photo_c)
        label_c.image = photo_c
        
        # Ảnh (d): Gaussian Thresholding
        frame_d = tk.Frame(row2, bg='#f0f0f0')
        frame_d.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_d, text="(d) Gaussian Thresholding", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_d, text="(ADAPTIVE_THRESH_GAUSSIAN_C)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_d = tk.Label(frame_d, bg='black', relief=tk.SUNKEN, bd=2)
        label_d.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_d_resized = self.resize_for_display(img_d, max_width=400, max_height=400)
        image_d = Image.fromarray(img_d_resized)
        photo_d = ImageTk.PhotoImage(image=image_d)
        label_d.config(image=photo_d)
        label_d.image = photo_d
    
    
    
    def apply_optimum_thresholding(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Vui lòng chọn ảnh trước!")
            return
        
        block_size = self.optimum_block_size.get()
        c = self.optimum_c.get()
        
        img_a, img_b, img_c, img_d, otsu_threshold = self.optimum_thresholding(
            self.original_image, block_size, c
        )
        self.show_optimum_thresholding_results_window(img_a, img_b, img_c, img_d, otsu_threshold, block_size, c)
    
    def show_optimum_thresholding_results_window(self, img_a, img_b, img_c, img_d, otsu_threshold, block_size, c):
        """Hiển thị kết quả optimum thresholding trong cửa sổ mới"""
        # Tạo cửa sổ mới với scroll
        optimum_window = tk.Toplevel(self.window)
        optimum_window.title("Optimum Thresholding - Kết quả")
        optimum_window.geometry("1600x900")
        optimum_window.configure(bg='#f0f0f0')
        
        # Container với scroll
        canvas = tk.Canvas(optimum_window, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(optimum_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mousewheel để scroll
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        # Bind khi chuột vào/ra khỏi cửa sổ
        optimum_window.bind("<Enter>", bind_mousewheel)
        optimum_window.bind("<Leave>", unbind_mousewheel)
        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_label = tk.Label(scrollable_frame, text="Optimum Thresholding (Local Thresholding)", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=(10, 20))
        
        # Thông tin về tham số
        info_text = f"Tham số: Otsu Threshold={otsu_threshold:.1f} (tự động), Block Size={block_size}, C={c}"
        info_label = tk.Label(scrollable_frame, text=info_text, 
                             font=('Arial', 10, 'italic'), bg='#f0f0f0', fg='gray')
        info_label.pack(pady=(0, 10))
        
        # Hàng 1: (a) Original và (b) Global thresholding
        row1 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row1.pack(fill=tk.X, pady=5)
        
        # Ảnh (a): Original Image
        frame_a = tk.Frame(row1, bg='#f0f0f0')
        frame_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_a, text="(a) Original Image", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_a, text="(sau Median Blur)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_a = tk.Label(frame_a, bg='black', relief=tk.SUNKEN, bd=2)
        label_a.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_a_resized = self.resize_for_display(img_a, max_width=400, max_height=400)
        image_a = Image.fromarray(img_a_resized)
        photo_a = ImageTk.PhotoImage(image=image_a)
        label_a.config(image=photo_a)
        label_a.image = photo_a
        
        # Ảnh (b): Global thresholding
        frame_b = tk.Frame(row1, bg='#f0f0f0')
        frame_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_b, text="(b) Global thresholding", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_b, text=f"(Otsu's method, v = {otsu_threshold:.1f})", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_b = tk.Label(frame_b, bg='black', relief=tk.SUNKEN, bd=2)
        label_b.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_b_resized = self.resize_for_display(img_b, max_width=400, max_height=400)
        image_b = Image.fromarray(img_b_resized)
        photo_b = ImageTk.PhotoImage(image=image_b)
        label_b.config(image=photo_b)
        label_b.image = photo_b
        
        # Hàng 2: (c) Local - Mean Thresholding và (d) Local - Gaussian Thresholding
        row2 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row2.pack(fill=tk.X, pady=5)
        
        # Ảnh (c): Local - Mean Thresholding
        frame_c = tk.Frame(row2, bg='#f0f0f0')
        frame_c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_c, text="(c) Local - Mean Thresholding", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_c, text="(ADAPTIVE_THRESH_MEAN_C)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_c = tk.Label(frame_c, bg='black', relief=tk.SUNKEN, bd=2)
        label_c.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_c_resized = self.resize_for_display(img_c, max_width=400, max_height=400)
        image_c = Image.fromarray(img_c_resized)
        photo_c = ImageTk.PhotoImage(image=image_c)
        label_c.config(image=photo_c)
        label_c.image = photo_c
        
        # Ảnh (d): Local - Gaussian Thresholding
        frame_d = tk.Frame(row2, bg='#f0f0f0')
        frame_d.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_d, text="(d) Local - Gaussian Thresholding", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_d, text="(ADAPTIVE_THRESH_GAUSSIAN_C)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_d = tk.Label(frame_d, bg='black', relief=tk.SUNKEN, bd=2)
        label_d.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_d_resized = self.resize_for_display(img_d, max_width=400, max_height=400)
        image_d = Image.fromarray(img_d_resized)
        photo_d = ImageTk.PhotoImage(image=image_d)
        label_d.config(image=photo_d)
        label_d.image = photo_d
    
    def image_sharpening_workflow(self, img, gamma=0.5):
        """
        Complete image sharpening workflow:
        (a) Original image
        (b) Laplacian of (a)
        (c) Sharpened image = (a) + (b)
        (d) Sobel gradient of (a)
        (e) Sobel image smoothed with 5×5 box filter
        (f) Mask image = (b) × (e)
        (g) Sharpened image = (a) + (f)
        (h) Final result = power-law transformation of (g)
        """
        
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.copy()
        
        # (a) 
        img_a = img_gray.astype(np.float32)
        
        # Laplacian of (a)
        laplacian = self.laplacian_filter(img_gray)
        
        laplacian_min = np.min(laplacian)
        laplacian_max = np.max(laplacian)
        if laplacian_max > laplacian_min:
            laplacian_scaled = ((laplacian - laplacian_min) / (laplacian_max - laplacian_min)) * 255.0
        else:
            laplacian_scaled = np.zeros_like(laplacian)
        
        # (d) Sobel gradient of (a)
        sobel = self.sobel_gradient(img_gray)
        
        sobel_min = np.min(sobel)
        sobel_max = np.max(sobel)
        if sobel_max > sobel_min:
            sobel_scaled = ((sobel - sobel_min) / (sobel_max - sobel_min)) * 255.0
        else:
            sobel_scaled = np.zeros_like(sobel)
        
        # (e) Sobel image smoothed with 5×5 box filter
        sobel_smoothed = self.box_filter_5x5(sobel_scaled.astype(np.uint8))
        sobel_smoothed = sobel_smoothed.astype(np.float32)
        
        # (f)  (b) × (e) 
        laplacian_norm = laplacian_scaled / 255.0
        sobel_norm = sobel_smoothed / 255.0
        mask = laplacian_norm * sobel_norm * 255.0
        
        # (g) (a) + (f)
        img_g = img_a + mask
        
        img_g = np.clip(img_g, 0, 255)
        
        # (h) Final 
        img_g_normalized = img_g / 255.0
        img_h = np.power(img_g_normalized, gamma) * 255.0
        img_h = np.clip(img_h, 0, 255)
        

        result = img_h.astype(np.uint8)
        
        if len(img.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def apply_negative(self):
        if self.original_image is not None:
            self.processed_image = self.neg_img(self.original_image)
            self.processed_title_label.config(text="Negative image")
            self.show_processed_image(self.processed_image)
    
    def apply_line_detection(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Vui lòng chọn ảnh trước!")
            return
        
        img_a, img_b, img_c, img_d = self.line_detection(self.original_image)
        self.show_line_detection_results_window(img_a, img_b, img_c, img_d)
    
    def show_line_detection_results_window(self, img_a, img_b, img_c, img_d):
        """Hiển thị kết quả line detection trong cửa sổ mới"""
        # Tạo cửa sổ mới với scroll
        line_window = tk.Toplevel(self.window)
        line_window.title("Line Detection - Kết quả")
        line_window.geometry("1600x900")
        line_window.configure(bg='#f0f0f0')
        
        # Container với scroll
        canvas = tk.Canvas(line_window, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(line_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mousewheel để scroll - chỉ khi chuột ở trong cửa sổ này
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        # Bind khi chuột vào/ra khỏi cửa sổ
        line_window.bind("<Enter>", bind_mousewheel)
        line_window.bind("<Leave>", unbind_mousewheel)
        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_label = tk.Label(scrollable_frame, text="Line Detection - Laplacian Operator", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=(10, 20))
        
        # Thông tin về quy trình
        info_text = "Quy trình: Gaussian Blur (3x3) → Threshold Binary (50) → Laplacian (CV_16S, ksize=3)"
        info_label = tk.Label(scrollable_frame, text=info_text, 
                             font=('Arial', 10, 'italic'), bg='#f0f0f0', fg='gray')
        info_label.pack(pady=(0, 10))
        
        # Hàng 1: (a) Binary Image và (b) Laplacian
        row1 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row1.pack(fill=tk.X, pady=5)
        
        # Ảnh (a): Binary Image
        frame_a = tk.Frame(row1, bg='#f0f0f0')
        frame_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_a, text="(a) Binary Image", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_a, text="(Gaussian Blur + Threshold)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_a = tk.Label(frame_a, bg='black', relief=tk.SUNKEN, bd=2)
        label_a.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_a_resized = self.resize_for_display(img_a, max_width=400, max_height=400)
        image_a = Image.fromarray(img_a_resized)
        photo_a = ImageTk.PhotoImage(image=image_a)
        label_a.config(image=photo_a)
        label_a.image = photo_a
        
        # Ảnh (b): Laplacian
        frame_b = tk.Frame(row1, bg='#f0f0f0')
        frame_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_b, text="(b) Laplacian Image", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_b, text="(shows positive/negative double-line effect)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_b = tk.Label(frame_b, bg='black', relief=tk.SUNKEN, bd=2)
        label_b.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_b_resized = self.resize_for_display(img_b, max_width=400, max_height=400)
        image_b = Image.fromarray(img_b_resized)
        photo_b = ImageTk.PhotoImage(image=image_b)
        label_b.config(image=photo_b)
        label_b.image = photo_b
        
        # Hàng 2: (c) Absolute Laplacian và (d) Positive Laplacian
        row2 = tk.Frame(scrollable_frame, bg='#f0f0f0')
        row2.pack(fill=tk.X, pady=5)
        
        # Ảnh (c): Absolute value
        frame_c = tk.Frame(row2, bg='#f0f0f0')
        frame_c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_c, text="(c) Absolute Value of Laplacian", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_c, text="(convertScaleAbs)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_c = tk.Label(frame_c, bg='black', relief=tk.SUNKEN, bd=2)
        label_c.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_c_resized = self.resize_for_display(img_c, max_width=400, max_height=400)
        image_c = Image.fromarray(img_c_resized)
        photo_c = ImageTk.PhotoImage(image=image_c)
        label_c.config(image=photo_c)
        label_c.image = photo_c
        
        # Ảnh (d): Positive values
        frame_d = tk.Frame(row2, bg='#f0f0f0')
        frame_d.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(frame_d, text="(d) Positive Values of Laplacian", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        tk.Label(frame_d, text="(img_lap > 0)", 
                font=('Arial', 9, 'italic'), bg='#f0f0f0', fg='gray').pack()
        label_d = tk.Label(frame_d, bg='black', relief=tk.SUNKEN, bd=2)
        label_d.pack(pady=5, fill=tk.BOTH, expand=True)
        
        img_d_resized = self.resize_for_display(img_d, max_width=400, max_height=400)
        image_d = Image.fromarray(img_d_resized)
        photo_d = ImageTk.PhotoImage(image=image_d)
        label_d.config(image=photo_d)
        label_d.image = photo_d
    
    def apply_transformation(self):
        if self.original_image is None:
            return
        
        img = self.original_image.copy()
        transform_type = self.selected_transform.get()
        
        transform_names = {
            "negative": "Negative image",
            "log": "Biến đổi Log",
            "piecewise": "Biến đổi Piecewise-Linear",
            "gamma": "Biến đổi Gamma",
            "average": "Làm trơn ảnh (lọc trung bình)",
            "gaussian": "Làm trơn ảnh (lọc Gauss)",
            "median": "Làm trơn ảnh (lọc trung vị)",
            "min": "Lọc Min",
            "max": "Lọc Max",
            "midpoint": "Lọc Midpoint",
            "histogram": "Cân bằng sáng dùng Histogram",
            "sharpening": "Image Sharpening Workflow",
            "binary": "Chuyển đổi ảnh nhị phân",
            "dilation": "Dilation (Giãn nở)",
            "erosion": "Erosion (Co)",
            "opening": "Opening (Mở)",
            "closing": "Closing (Đóng)"
        }
        
        if transform_type == "negative":
            img = self.neg_img(img)
            
        elif transform_type == "log":
            c = self.log_c_slider.get()
            img = self.log_transform(img, c)
            
        elif transform_type == "piecewise":
            low = self.piecewise_low.get()
            high = self.piecewise_high.get()
            img = self.piecewise_linear(img, low, high)
            
        elif transform_type == "gamma":
            gamma_c = self.gamma_c.get()
            gamma = self.gamma_value.get()
            img = self.gamma_transform(img, gamma_c, gamma)
            
        elif transform_type == "average":
            avg_k = self.avg_kernel.get()
            img = self.average_filter(img, avg_k)
            
        elif transform_type == "gaussian":
            gauss_k = self.gauss_kernel.get()
            gauss_s = self.gauss_sigma.get()
            img = self.gaussian_filter(img, gauss_k, gauss_s)
            
        elif transform_type == "median":
            med_k = self.median_kernel.get()
            img = self.median_filter(img, med_k)
            
        elif transform_type == "min":
            min_k = self.min_kernel.get()
            img = self.min_filter(img, min_k)
            
        elif transform_type == "max":
            max_k = self.max_kernel.get()
            img = self.max_filter(img, max_k)
            
        elif transform_type == "midpoint":
            mid_k = self.midpoint_kernel.get()
            img = self.midpoint_filter(img, mid_k)
            
        elif transform_type == "histogram":
            img = self.histogram_equalization(img)
        
        elif transform_type == "sharpening":
            gamma = self.sharpening_gamma.get()
            img = self.image_sharpening_workflow(img, gamma)
        
        elif transform_type == "binary":
            threshold = self.binary_threshold.get()
            img = self.convert_to_binary(img, threshold=threshold)
            # Chuyển đổi sang RGB để hiển thị (vì ảnh nhị phân là grayscale)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        elif transform_type == "dilation":
            kernel_size = self.dilation_kernel.get()
            iterations = self.dilation_iterations.get()
            img = self.delation(img, kernel_size=kernel_size, iterations=iterations)
        
        elif transform_type == "erosion":
            kernel_size = self.erosion_kernel.get()
            iterations = self.erosion_iterations.get()
            img = self.erosion(img, kernel_size=kernel_size, iterations=iterations)
        
        elif transform_type == "opening":
            kernel_size = self.opening_kernel.get()
            img = self.opening(img, kernel_size=kernel_size)
        
        elif transform_type == "closing":
            kernel_size = self.closing_kernel.get()
            img = self.closing(img, kernel_size=kernel_size)
        
        self.processed_title_label.config(text=transform_names.get(transform_type, "Processed image"))
        self.processed_image = img
        self.show_processed_image(self.processed_image)
    
    def apply_fourier_transform(self):
        """Áp dụng biến đổi Fourier và hiển thị 3 ảnh: Input, Spectrum, IFFT"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Vui lòng chọn ảnh trước!")
            return
        
        # Đo thời gian tổng
        total_start = time.time()
        
        # Áp dụng biến đổi Fourier với đo thời gian
        magnitude_spectrum, F_uv_shifted, time_info, channels_magnitude = self.spatial_to_frequency_domain(self.original_image, measure_time=True)
        
        # Chuyển sang RGB để hiển thị (vì magnitude_spectrum là grayscale)
        if len(magnitude_spectrum.shape) == 2:
            magnitude_spectrum_rgb = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)
        else:
            magnitude_spectrum_rgb = magnitude_spectrum
        
        # Tính IFFT để chuyển ngược về miền không gian
        ifft_image, ifft_time_info = self.frequency_to_spatial_domain(F_uv_shifted, measure_time=True)
        
        total_time = (time.time() - total_start) * 1000  # ms
        
        # Hiển thị ảnh trong cửa sổ mới (bao gồm từng kênh nếu là ảnh màu)
        self.show_fourier_results_window(self.original_image, magnitude_spectrum_rgb, ifft_image, channels_magnitude)
        
        # Xuất thời gian ra console
        print("\n" + "="*50)
        print("BIẾN ĐỔI FOURIER - THỜI GIAN THỰC THI")
        print("="*50)
        print(f"Tổng thời gian: {total_time:.3f} ms")
        
        # Thời gian FFT (chuyển từ miền không gian sang miền tần số)
        print("\n--- FFT: Chuyển từ miền không gian sang miền tần số ---")
        fft_total = sum(time_info.values())
        print(f"Tổng thời gian FFT: {fft_total:.3f} ms")
        for step, step_time in time_info.items():
            print(f"  {step}: {step_time:.3f} ms")
        
        # Thời gian IFFT (chuyển ngược về miền không gian)
        print("\n--- IFFT: Chuyển ngược về miền không gian ---")
        ifft_total = sum(ifft_time_info.values())
        print(f"Tổng thời gian IFFT: {ifft_total:.3f} ms")
        for step, step_time in ifft_time_info.items():
            print(f"  {step}: {step_time:.3f} ms")
        
        print("="*50 + "\n")
    
    def analyze_binary_image_action(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Vui lòng chọn ảnh trước!")
            return
        
        img_to_analyze = self.processed_image if self.processed_image is not None else self.original_image
        
        result = self.analyze_binary_image(img_to_analyze)
        
        areas_str = "\n".join([f"  Đối tượng {i+1}: {area} pixels" 
                               for i, area in enumerate(result['object_areas'])])
        
        message = f"Số lượng đối tượng: {result['num_objects']}\n\n"
        message += f"Diện tích từng đối tượng (pixels):\n{areas_str}\n\n"
        message += "(Xem console để biết thêm chi tiết)"
        
        messagebox.showinfo("Kết quả phân tích ảnh nhị phân", message)
        
        if result['num_objects'] > 0:
            labels_colored = self.colorize_labels(result['labels'])
            self.processed_image = labels_colored
            self.processed_title_label.config(text="Ảnh đã gán nhãn (mỗi đối tượng một màu)")
            self.show_processed_image(self.processed_image)
    
    def colorize_labels(self, labels):
        num_labels = len(np.unique(labels))
        h, w = labels.shape
        
        np.random.seed(42)  
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  
        
        # Tạo ảnh màu
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(num_labels):
            colored[labels == i] = colors[i]
        
        return colored
    
    def update_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to update")
            return
        
        self.original_image = self.processed_image.copy()
        self.show_original_image()
        self.apply_transformation()
        
        messagebox.showinfo("Success", "Ảnh đã được cập nhật làm ảnh gốc mới!")
    
    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            img_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, img_bgr)
            messagebox.showinfo("Success", "Image saved successfully")


def main():
    app = UI()
    app.window.mainloop()

if __name__ == "__main__":
    main()