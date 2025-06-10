"""
Enhanced T2 Mapping Tool with Additional Diagnostics
Fixes potential issues with T2 decay not being observed
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import interp1d
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

class T2MappingApp:
    """Main application class for T2 mapping with T2prep sequences"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("T2 Mapping Tool - Enhanced with Diagnostics")
        self.root.geometry("1200x700")
        
        # Initialize variables
        self.dicom_files = []
        self.image_stack = None
        self.te_preps = None  # T2 preparation times
        self.corrected_stack = None
        self.t2_map = None
        self.t2_map_full = None
        self.r2_map = None
        self.roi_coords = None
        self.pixel_cache = {}
        self.b1_map = None
        self.noise_level = None
        self.dicom_headers = []  # Store DICOM headers for debugging
        
        # Create UI
        self.create_ui()
        self.create_menu()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()
        print(f"Status: {message}")  # Also print to console

    def create_ui(self):
        """Create the main user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        self.left_panel = ttk.LabelFrame(main_frame, text="Controls", padding=10, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_panel.pack_propagate(False)
        
        # Right panel for visualization
        self.right_panel = ttk.LabelFrame(main_frame, text="Visualization", padding=10)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create control panels
        self.create_load_panel()
        self.create_correction_panel()
        self.create_processing_panel()
        self.create_visualization_panel()
        
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Load T2prep DICOM files to begin", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        self.canvas.draw()
        
    def create_load_panel(self):
        """Create data loading panel"""
        load_frame = ttk.LabelFrame(self.left_panel, text="Data Loading", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(load_frame, text="Load T2prep DICOM Files", 
                  command=self.load_dicom_files).pack(fill=tk.X, pady=5)
        
        # Add diagnostic button
        self.diag_button = ttk.Button(load_frame, text="Run Diagnostics", 
                                     command=self.run_comprehensive_diagnostics,
                                     state=tk.DISABLED)
        self.diag_button.pack(fill=tk.X, pady=5)
        
        self.file_info_var = tk.StringVar(value="No files loaded")
        ttk.Label(load_frame, textvariable=self.file_info_var, 
                 wraplength=250).pack(fill=tk.X)

    def create_correction_panel(self):
        """Create correction options panel"""
        corr_frame = ttk.LabelFrame(self.left_panel, text="Correction Options", padding=10)
        corr_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(corr_frame, text="Sequence Type:").pack(anchor=tk.W, pady=(10, 0))
        self.sequence_type_var = tk.StringVar(value="bSSFP")
        sequence_frame = ttk.Frame(corr_frame)
        sequence_frame.pack(fill=tk.X)
        ttk.Radiobutton(sequence_frame, text="bSSFP", 
                       variable=self.sequence_type_var, 
                       value="bSSFP").pack(side=tk.LEFT)
        ttk.Radiobutton(sequence_frame, text="GRE", 
                       variable=self.sequence_type_var, 
                       value="GRE").pack(side=tk.LEFT)
        # Noise floor correction
        self.noise_correction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(corr_frame, text="Noise Floor Correction", 
                       variable=self.noise_correction_var).pack(anchor=tk.W, pady=2)
        
        # Signal normalization (NEW)
        self.normalize_signals_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(corr_frame, text="Normalize Signal Intensities", 
                       variable=self.normalize_signals_var).pack(anchor=tk.W, pady=2)
        
        # B1 correction
        self.b1_correction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(corr_frame, text="B1 Inhomogeneity Correction", 
                       variable=self.b1_correction_var).pack(anchor=tk.W, pady=2)
        
        # Stimulated echo correction
        self.stim_echo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(corr_frame, text="Stimulated Echo Correction", 
                       variable=self.stim_echo_var).pack(anchor=tk.W, pady=2)
        
        # T1 contamination correction
        self.t1_correction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(corr_frame, text="T1 Contamination Correction", 
                       variable=self.t1_correction_var).pack(anchor=tk.W, pady=2)
        
        # T1 value input
        t1_frame = ttk.Frame(corr_frame)
        t1_frame.pack(fill=tk.X, pady=5)
        ttk.Label(t1_frame, text="T1 (ms):").pack(side=tk.LEFT, padx=5)
        self.t1_value_var = tk.StringVar(value="1200")
        ttk.Entry(t1_frame, textvariable=self.t1_value_var, width=10).pack(side=tk.LEFT)
        
        # Motion correction
        self.motion_correction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(corr_frame, text="Motion Correction", 
                       variable=self.motion_correction_var).pack(anchor=tk.W, pady=2)
        
        # Fitting method
        ttk.Label(corr_frame, text="Fitting Method:").pack(anchor=tk.W, pady=(10, 0))
        self.fitting_method_var = tk.StringVar(value="3-parameter")
        methods_frame = ttk.Frame(corr_frame)
        methods_frame.pack(fill=tk.X)
        ttk.Radiobutton(methods_frame, text="2-parameter", 
                       variable=self.fitting_method_var, 
                       value="2-parameter").pack(side=tk.LEFT)
        ttk.Radiobutton(methods_frame, text="3-parameter", 
                       variable=self.fitting_method_var, 
                       value="3-parameter").pack(side=tk.LEFT)

    def create_processing_panel(self):
        """Create processing panel"""
        proc_frame = ttk.LabelFrame(self.left_panel, text="T2 Mapping", padding=10)
        proc_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.calc_t2_button = ttk.Button(proc_frame, text="Select ROI & Calculate T2", 
                                        command=self.calculate_t2,
                                        state=tk.DISABLED)
        self.calc_t2_button.pack(fill=tk.X, pady=5)
        
        self.calc_full_button = ttk.Button(proc_frame, text="Calculate Full T2 Map", 
                                          command=self.calculate_full_t2_map,
                                          state=tk.DISABLED)
        self.calc_full_button.pack(fill=tk.X, pady=5)
        
        self.t2_result_var = tk.StringVar(value="No T2 calculated")
        ttk.Label(proc_frame, textvariable=self.t2_result_var, 
                 wraplength=250).pack(fill=tk.X)
        
        # Export buttons
        export_frame = ttk.LabelFrame(self.left_panel, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.save_button = ttk.Button(export_frame, text="Save T2 Map", 
                                     command=self.save_t2_map,
                                     state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=2)

    def create_visualization_panel(self):
        """Create visualization options panel"""
        viz_frame = ttk.LabelFrame(self.left_panel, text="Visualization", padding=10)
        viz_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Colormap selection
        ttk.Label(viz_frame, text="Colormap:").grid(row=0, column=0, sticky='w')
        self.colormap_var = tk.StringVar(value='viridis')
        colormap_combo = ttk.Combobox(viz_frame, textvariable=self.colormap_var,
                                     values=['viridis', 'jet', 'hot', 'cool', 'turbo'],
                                     width=15)
        colormap_combo.grid(row=0, column=1, padx=5)
        colormap_combo.bind("<<ComboboxSelected>>", self.update_display)
        
        # T2 range
        ttk.Label(viz_frame, text="T2 Range (ms):").grid(row=1, column=0, sticky='w', pady=5)
        range_frame = ttk.Frame(viz_frame)
        range_frame.grid(row=1, column=1, padx=5, pady=5)
        
        self.t2_min_var = tk.StringVar(value="0")
        self.t2_max_var = tk.StringVar(value="150")
        ttk.Entry(range_frame, textvariable=self.t2_min_var, width=5).pack(side=tk.LEFT)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT, padx=2)
        ttk.Entry(range_frame, textvariable=self.t2_max_var, width=5).pack(side=tk.LEFT)
        
        ttk.Button(viz_frame, text="Update Display", 
                  command=self.update_display).grid(row=2, column=0, columnspan=2, pady=5)
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load DICOM Files", command=self.load_dicom_files)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Run Diagnostics", command=self.run_comprehensive_diagnostics)
        tools_menu.add_command(label="Check DICOM Headers", command=self.check_dicom_headers)

    def load_dicom_files(self):
        """Load T2prep DICOM files with enhanced error checking"""
        try:
            files = filedialog.askopenfilenames(
                title="Select T2prep DICOM Files",
                filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
            )
            
            if not files:
                return
                
            self.update_status(f"Loading {len(files)} files...")
            
            # Load and sort by TE prep time
            self.image_stack, self.te_preps, self.dicom_headers = self.load_t2prep_dicoms_enhanced(files)
            self.dicom_files = files
            
            # Check for valid TE prep times
            if len(set(self.te_preps)) < 2:
                messagebox.showwarning("Warning", 
                    "Less than 2 unique TE prep times detected. T2 fitting may not work properly.\n"
                    "Please check if the correct DICOM files were loaded.")
            
            # Estimate noise level
            self.estimate_noise_level()
            
            # Apply normalization if needed
            #if self.normalize_signals_var.get():
            #    self.normalize_signal_intensities()
            
            # Apply motion correction if enabled
            if self.motion_correction_var.get():
                self.update_status("Applying motion correction...")
                self.corrected_stack = self.apply_motion_correction(self.image_stack)
            else:
                self.corrected_stack = self.image_stack.copy()
            
            # Update UI
            self.file_info_var.set(f"Loaded {len(files)} files\n"
                                 f"TE prep times: {', '.join([f'{te:.1f}' for te in self.te_preps])} ms\n"
                                 f"Image size: {self.image_stack.shape[1]}×{self.image_stack.shape[2]} px\n"
                                 f"Noise level: {self.noise_level:.1f}")
            
            self.calc_t2_button.configure(state=tk.NORMAL)
            self.calc_full_button.configure(state=tk.NORMAL)
            self.diag_button.configure(state=tk.NORMAL)
            
            # Show image overview
            self.show_image_overview()
            
            # Automatically run diagnostics
            self.diagnose_t2_data()
            
            self.update_status("Files loaded successfully")
            
        except Exception as e:
            self.update_status("Error loading files")
            messagebox.showerror("Error", f"Error loading DICOM files: {str(e)}")

    def load_t2prep_dicoms_enhanced(self, file_paths):
        """Enhanced DICOM loading with better TE prep extraction"""
        images = []
        te_preps = []
        headers = []
        
        for i, path in enumerate(file_paths):
            print(f"\nLoading file {i+1}/{len(file_paths)}: {os.path.basename(path)}")
            
            ds = pydicom.dcmread(path)
            headers.append(ds)
            
            # Extract pixel data
            img = ds.pixel_array.astype(np.float32)
            
            # Apply rescale if available
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            img = img * slope + intercept
            
            # Window/Level adjustment if specified
            #if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            #    wc = float(ds.WindowCenter)
            #    ww = float(ds.WindowWidth)
            #    img = np.clip(img, wc - ww/2, wc + ww/2)
            
            images.append(img)
            
            # Enhanced TE prep extraction
            te_prep = self.extract_te_prep_enhanced(ds, i)
            te_preps.append(te_prep)
            
            print(f"  Image shape: {img.shape}")
            print(f"  Signal range: [{np.min(img):.1f}, {np.max(img):.1f}]")
            print(f"  TE prep: {te_prep:.1f} ms")
        
        # Verify TE prep times
        print(f"\nExtracted TE prep times: {te_preps}")
        
        # Check if we got valid T2prep times
        if all(te == 0.0 for te in te_preps) or len(set(te_preps)) < 2:
            print("\nWARNING: Could not extract valid TE prep times from DICOM headers!")
            te_preps = self.get_manual_te_preps(len(images))
            if te_preps is None:
                raise ValueError("T2prep times are required for processing")
        
        # Sort by TE prep time
        sorted_indices = np.argsort(te_preps)
        images = [images[i] for i in sorted_indices]
        te_preps = [te_preps[i] for i in sorted_indices]
        headers = [headers[i] for i in sorted_indices]
        
        print(f"\nFinal sorted TE prep times: {te_preps}")
        
        return np.stack(images, axis=0), np.array(te_preps), headers

    def extract_te_prep_enhanced(self, ds, file_index):
        """Enhanced extraction of T2 preparation time from DICOM"""
        te_prep = 0.0
        
        # 1. Check Echo Time (but be careful about readout TE vs prep TE)
        if hasattr(ds, 'EchoTime'):
            te = float(ds.EchoTime)
            # T2prep times are typically > 20ms, readout TE is typically < 10ms
            if te > 20:
                te_prep = te
                print(f"  Found TE prep in EchoTime: {te_prep:.1f} ms")
                return te_prep
        
        # 2. Check Effective Echo Time
        if hasattr(ds, 'EffectiveEchoTime'):
            te_prep = float(ds.EffectiveEchoTime)
            print(f"  Found TE prep in EffectiveEchoTime: {te_prep:.1f} ms")
            return te_prep
        
        # 3. Check sequence-specific private tags
        # Siemens
        siemens_tags = [(0x0019, 0x1013), (0x0019, 0x1012), (0x0019, 0x1018)]
        for tag in siemens_tags:
            if tag in ds:
                try:
                    te_prep = float(ds[tag].value)
                    if te_prep > 0:
                        print(f"  Found TE prep in Siemens private tag {tag}: {te_prep:.1f} ms")
                        return te_prep
                except:
                    pass
        
        # 4. Check Philips private tags
        philips_tags = [(0x2001, 0x1008), (0x2001, 0x1060)]
        for tag in philips_tags:
            if tag in ds:
                try:
                    te_prep = float(ds[tag].value)
                    if te_prep > 0:
                        print(f"  Found TE prep in Philips private tag {tag}: {te_prep:.1f} ms")
                        return te_prep
                except:
                    pass
        
        # 5. Parse from sequence/series description
        descriptions = []
        if hasattr(ds, 'SeriesDescription'):
            descriptions.append(str(ds.SeriesDescription))
        if hasattr(ds, 'SequenceName'):
            descriptions.append(str(ds.SequenceName))
        if hasattr(ds, 'ProtocolName'):
            descriptions.append(str(ds.ProtocolName))
            
        import re
        for desc in descriptions:
            # Look for patterns like "T2prep50", "T2_50", "TE50", "TEprep50"
            patterns = [
                r'T2[_\s]*prep[_\s]*(\d+)',
                r'TE[_\s]*prep[_\s]*(\d+)',
                r'T2[_\s]*(\d+)',
                r'TE[_\s]*(\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, desc, re.IGNORECASE)
                if match:
                    te_prep = float(match.group(1))
                    if te_prep > 0:
                        print(f"  Found TE prep in {desc}: {te_prep:.1f} ms")
                        return te_prep
        
        # 6. Check for instance number pattern (sometimes files are numbered by TE)
        if hasattr(ds, 'InstanceNumber'):
            # Common patterns: 0ms=instance 1, 25ms=instance 2, etc.
            instance = int(ds.InstanceNumber)
            # This is a guess - adjust based on your protocol
            possible_te = (instance - 1) * 25.0
            if file_index == instance - 1:  # Verify ordering makes sense
                print(f"  Guessing TE prep from instance number: {possible_te:.1f} ms")
                return possible_te
        
        print(f"  WARNING: Could not extract TE prep time!")
        return 0.0

    def identify_noise_regions(self):
        """
        Identify true noise/background regions using image statistics
        Returns a binary mask of noise regions
        """
        # Use first image to identify background
        img = self.image_stack[0]
        
        # Method: Otsu thresholding to separate signal from noise
        from skimage.filters import threshold_otsu
        
        # Apply Gaussian filter to reduce noise
        from scipy.ndimage import gaussian_filter
        img_smooth = gaussian_filter(img, sigma=2)
        
        # Find threshold
        try:
            thresh = threshold_otsu(img_smooth)
            noise_mask = img < (thresh * 0.5)  # Conservative: well below threshold
            
            # Ensure we have enough noise pixels
            noise_pixel_count = np.sum(noise_mask)
            total_pixels = img.size
            
            if noise_pixel_count > 0.01 * total_pixels:  # At least 1% of image
                # Additional check: noise regions should have low mean and low variance
                noise_mean = np.mean(img[noise_mask])
                signal_mean = np.mean(img[~noise_mask])
                
                if noise_mean < 0.2 * signal_mean:  # Noise is much lower than signal
                    return noise_mask
        except:
            pass
        
        return None

    def find_stable_reference_tissue(self):
        """
        Find tissue region with long T2 (minimal decay) for normalization
        Suitable for cardiac or musculoskeletal imaging
        """
        if len(self.image_stack) < 3:
            return None
        
        # Calculate decay map
        first_img = self.image_stack[0]
        last_img = self.image_stack[-1]
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            decay_map = last_img / (first_img + 1e-10)
        
        # Look for regions with minimal decay (high decay_map values)
        # Apply smoothing to reduce noise
        from scipy.ndimage import gaussian_filter
        decay_map_smooth = gaussian_filter(decay_map, sigma=3)
        
        # Find regions with decay > 0.7 (less than 30% signal loss)
        stable_mask = (decay_map_smooth > 0.7) & (first_img > 3 * self.noise_level)
        
        # Find largest connected component
        from scipy.ndimage import label
        labeled, num_features = label(stable_mask)
        
        if num_features > 0:
            # Get the largest stable region
            component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest_component = np.argmax(component_sizes) + 1
            
            reference_mask = labeled == largest_component
            
            # Ensure it's large enough
            if np.sum(reference_mask) > 100:  # At least 100 pixels
                return reference_mask
        
        return None

    def fit_reference_t2(self, te_preps, signals):
        """
        Fit T2 to reference tissue signals
        """
        try:
            # Simple exponential fit
            def exp_func(t, s0, t2):
                return s0 * np.exp(-t / t2)
            
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(exp_func, te_preps, signals, 
                               p0=[signals[0], 200])  # Assume long T2
            
            return popt[1]  # Return T2
        except:
            return 200.0  # Default long T2


    def robust_std(self, data):
        """
        Calculate robust standard deviation using MAD
        (Median Absolute Deviation)
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        # Convert MAD to standard deviation equivalent
        return 1.4826 * mad


    # Alternative: Completely disable normalization for certain protocols
    def should_apply_normalization(self):
        """
        Determine if normalization should be applied based on protocol
        """
        # Check if this is a standard T2 mapping protocol
        if self.dicom_headers:
            # Check for specific sequence names that shouldn't be normalized
            for ds in self.dicom_headers:
                if hasattr(ds, 'SeriesDescription'):
                    desc = str(ds.SeriesDescription).lower()
                    # Skip normalization for certain protocols
                    if any(keyword in desc for keyword in ['t2_map', 'cmr_mapping', 't2prep']):
                        if 'normalized' not in desc:  # Unless explicitly normalized
                            print("Standard T2 mapping protocol detected - skipping normalization")
                            return False
        
        return True


    # Method to validate normalization effects
    def validate_normalization(self, before_stack, after_stack):
        """
        Check if normalization preserves T2 decay characteristics
        """
        # Sample several ROIs
        h, w = before_stack.shape[1:]
        test_rois = [
            (h//2, w//2, 50),    # Center
            (h//3, w//3, 30),    # Upper left
            (2*h//3, 2*w//3, 30) # Lower right
        ]
        
        print("\nNormalization validation:")
        
        for roi_y, roi_x, roi_size in test_rois:
            # Extract signals before and after
            before_signals = []
            after_signals = []
            
            for i in range(len(before_stack)):
                roi_before = before_stack[i, 
                                        roi_y-roi_size//2:roi_y+roi_size//2,
                                        roi_x-roi_size//2:roi_x+roi_size//2]
                roi_after = after_stack[i,
                                      roi_y-roi_size//2:roi_y+roi_size//2,
                                      roi_x-roi_size//2:roi_x+roi_size//2]
                
                before_signals.append(np.mean(roi_before))
                after_signals.append(np.mean(roi_after))
            
            # Fit T2 to both
            t2_before = self.quick_t2_fit(self.te_preps, before_signals)
            t2_after = self.quick_t2_fit(self.te_preps, after_signals)
            
            print(f"  ROI at ({roi_x}, {roi_y}): T2 before={t2_before:.1f}ms, "
                  f"after={t2_after:.1f}ms, change={100*(t2_after-t2_before)/t2_before:.1f}%")
            
            # Warn if T2 changes significantly
            if abs(t2_after - t2_before) / t2_before > 0.1:  # >10% change
                print("  WARNING: Normalization significantly altered T2 values!")


    def quick_t2_fit(self, te_preps, signals):
        """Quick T2 estimation for validation"""
        try:
            # Log-linear fit
            valid = np.array(signals) > 0
            if np.sum(valid) >= 2:
                log_signals = np.log(np.array(signals)[valid])
                te_valid = np.array(te_preps)[valid]
                slope, _ = np.polyfit(te_valid, log_signals, 1)
                return -1.0 / slope if slope < 0 else 100.0
        except:
            pass
        return 50.0  # Default
    
    def check_dicom_headers(self):
        """Display DICOM header information for debugging"""
        if not self.dicom_headers:
            messagebox.showinfo("Info", "No DICOM files loaded")
            return
            
        # Create window for header display
        header_window = tk.Toplevel(self.root)
        header_window.title("DICOM Header Information")
        header_window.geometry("800x600")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(header_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Display header information
        for i, (ds, te_prep) in enumerate(zip(self.dicom_headers, self.te_preps)):
            text_widget.insert(tk.END, f"\n{'='*60}\n")
            text_widget.insert(tk.END, f"FILE {i+1}: TE prep = {te_prep:.1f} ms\n")
            text_widget.insert(tk.END, f"{'='*60}\n\n")
            
            # Key fields to display
            key_fields = [
                'SeriesDescription', 'ProtocolName', 'SequenceName',
                'EchoTime', 'EffectiveEchoTime', 'RepetitionTime',
                'FlipAngle', 'InstanceNumber', 'AcquisitionNumber',
                'TemporalPositionIdentifier', 'InversionTime'
            ]
            
            for field in key_fields:
                if hasattr(ds, field):
                    value = getattr(ds, field)
                    text_widget.insert(tk.END, f"{field}: {value}\n")
            
            # Check for relevant private tags
            text_widget.insert(tk.END, "\nPrivate Tags:\n")
            for elem in ds:
                if elem.tag.is_private and elem.VR != 'UN':
                    if 0x0019 <= elem.tag.group <= 0x0021 or elem.tag.group == 0x2001:
                        try:
                            text_widget.insert(tk.END, f"  {elem.tag}: {elem.value}\n")
                        except:
                            pass
        
        text_widget.config(state=tk.DISABLED)

    

    def diagnose_t2_data(self):
        """Enhanced diagnostic function to check T2 decay curves"""
        if self.corrected_stack is None:
            print("No data loaded")
            return
            
        # Create diagnostic plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("T2 Data Diagnostics", fontsize=16)
        
        # Plot 1: TE prep times
        ax1 = axes[0, 0]
        ax1.bar(range(len(self.te_preps)), self.te_preps)
        ax1.set_xlabel("Image Index")
        ax1.set_ylabel("TE prep (ms)")
        ax1.set_title("TE Preparation Times")
        for i, te in enumerate(self.te_preps):
            ax1.text(i, te + 2, f'{te:.0f}', ha='center', fontsize=8)
        
        # Plot 2: Mean signal vs TE prep
        ax2 = axes[0, 1]
        mean_signals = [np.mean(img) for img in self.corrected_stack]
        ax2.scatter(self.te_preps, mean_signals, s=100, c='blue', label='Mean')
        ax2.plot(self.te_preps, mean_signals, 'b--', alpha=0.5)
        
        # Add exponential fit
        if len(self.te_preps) > 2:
            try:
                # Fit exponential
                popt, _ = opt.curve_fit(lambda t, a, b: a * np.exp(-t/b), 
                                      self.te_preps, mean_signals,
                                      p0=[mean_signals[0], 50])
                t_fit = np.linspace(0, max(self.te_preps), 100)
                ax2.plot(t_fit, popt[0] * np.exp(-t_fit/popt[1]), 'r-', 
                        label=f'Fit: T2={popt[1]:.1f}ms')
                ax2.legend()
            except:
                pass
        
        ax2.set_xlabel("TE prep (ms)")
        ax2.set_ylabel("Mean Signal")
        ax2.set_title("Mean Signal Decay")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Multiple ROI signals
        ax3 = axes[0, 2]
        
        # Sample multiple ROIs
        h, w = self.corrected_stack.shape[1:]
        roi_size = 20
        roi_positions = [
            (h//2, w//2, 'Center'),
            (h//4, w//4, 'Top-Left'),
            (3*h//4, 3*w//4, 'Bottom-Right'),
        ]
        
        colors = ['red', 'green', 'blue']
        for (cy, cx, label), color in zip(roi_positions, colors):
            roi_signals = []
            for img in self.corrected_stack:
                roi = img[cy-roi_size//2:cy+roi_size//2, 
                         cx-roi_size//2:cx+roi_size//2]
                roi_signals.append(np.mean(roi))
            
            # Normalize to first value for comparison
            roi_signals_norm = np.array(roi_signals) / roi_signals[0]
            ax3.scatter(self.te_preps, roi_signals_norm, s=50, 
                       color=color, label=label)
            ax3.plot(self.te_preps, roi_signals_norm, '--', 
                    color=color, alpha=0.5)
        
        ax3.set_xlabel("TE prep (ms)")
        
    
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Signal histogram
        ax4 = axes[1, 0]
        for i, (img, te) in enumerate(zip(self.corrected_stack, self.te_preps)):
            hist, bins = np.histogram(img.flatten(), bins=50)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax4.plot(bin_centers, hist, label=f'TE={te:.0f}ms', alpha=0.7)
        ax4.set_xlabel("Signal Intensity")
        ax4.set_ylabel("Count")
        ax4.set_title("Signal Histograms")
        ax4.legend()
        ax4.set_yscale('log')
        
        # Plot 5: SNR vs TE
        ax5 = axes[1, 1]
        snr_values = []
        for img in self.corrected_stack:
            # Estimate signal from center, noise from corners
            signal = np.mean(img[h//2-50:h//2+50, w//2-50:w//2+50])
            noise = np.std(img[:20, :20])
            snr = signal / noise if noise > 0 else 0
            snr_values.append(snr)
        
        ax5.scatter(self.te_preps, snr_values, s=100, c='purple')
        ax5.plot(self.te_preps, snr_values, 'purple', linestyle='--', alpha=0.5)
        ax5.set_xlabel("TE prep (ms)")
        ax5.set_ylabel("SNR")
        ax5.set_title("Signal-to-Noise Ratio")
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Expected vs actual decay
        ax6 = axes[1, 2]
        
        # Normalize signals to first value
        center_roi = []
        cy, cx = h//2, w//2
        for img in self.corrected_stack:
            roi = img[cy-roi_size//2:cy+roi_size//2, 
                     cx-roi_size//2:cx+roi_size//2]
            center_roi.append(np.mean(roi))
        
        if center_roi[0] > 0:
            actual_ratios = np.array(center_roi) / center_roi[0]
            ax6.scatter(self.te_preps, actual_ratios, s=100, 
                       label='Measured', color='blue')
            
            # Plot expected curves for different T2 values
            te_range = np.linspace(0, max(self.te_preps), 100)
            for t2 in [30, 50, 70, 100]:
                expected = np.exp(-te_range / t2)
                ax6.plot(te_range, expected, '--', alpha=0.5, 
                        label=f'T2={t2}ms')
        
        ax6.set_xlabel("TE prep (ms)")
        ax6.set_ylabel("Signal Ratio (S/S0)")
        ax6.set_title("Measured vs Expected Decay")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.show()
        
        # Print diagnostic report
        self.print_diagnostic_report(mean_signals, snr_values)

    def print_diagnostic_report(self, mean_signals, snr_values):
        """Print detailed diagnostic report"""
        print("\n" + "="*70)
        print("T2 DATA DIAGNOSTIC REPORT")
        print("="*70)
        print(f"Number of images: {len(self.te_preps)}")
        print(f"TE prep times: {self.te_preps}")
        print(f"Image shape: {self.corrected_stack.shape}")
        print(f"Noise level: {self.noise_level:.2f}")
        
        print("\nSignal characteristics:")
        for i, (te, signal, snr) in enumerate(zip(self.te_preps, mean_signals, snr_values)):
            print(f"  Image {i}: TE={te:.1f}ms")
            print(f"    Mean signal: {signal:.1f}")
            print(f"    Signal range: [{np.min(self.corrected_stack[i]):.1f}, "
                  f"{np.max(self.corrected_stack[i]):.1f}]")
            print(f"    SNR: {snr:.1f}")
        
        # Check for proper decay
        if len(self.te_preps) > 1 and mean_signals[0] > 0:
            signal_ratios = [s/mean_signals[0] for s in mean_signals]
            print(f"\nSignal decay ratios (relative to first image):")
            for te, ratio in zip(self.te_preps, signal_ratios):
                print(f"  TE={te:.1f}ms: {ratio:.3f}")
            
            # Check if decay is monotonic
            is_monotonic = all(signal_ratios[i] >= signal_ratios[i+1] 
                             for i in range(len(signal_ratios)-1))
            
            if not is_monotonic:
                print("\nWARNING: Signal decay is not monotonic!")
                print("Possible issues:")
                print("  - Images may be in wrong order")
                print("  - TE prep times may be incorrect")
                print("  - Signal scaling issues between images")
            else:
                print("\n✓ Signal decay appears monotonic")
                
            # Estimate apparent T2
            if len(self.te_preps) >= 2:
                try:
                    # Simple exponential fit
                    popt, _ = opt.curve_fit(
                        lambda t, a, b: a * np.exp(-t/b),
                        self.te_preps, mean_signals,
                        p0=[mean_signals[0], 50]
                    )
                    print(f"\nApproximate T2 from mean signals: {popt[1]:.1f} ms")
                except:
                    print("\nCould not estimate T2 from mean signals")
        
        print("="*70)

    def run_comprehensive_diagnostics(self):
        """Run comprehensive diagnostics including signal analysis"""
        if self.corrected_stack is None:
            messagebox.showinfo("Info", "Please load DICOM files first")
            return
            
        # Run basic diagnostics
        self.diagnose_t2_data()
        
        # Additional diagnostic: Check individual pixel decays
        self.check_pixel_decays()

    def check_pixel_decays(self):
        """Check individual pixel decay behaviors"""
        if self.corrected_stack is None:
            return
            
        h, w = self.corrected_stack.shape[1:]
        
        # Sample random pixels
        n_samples = 100
        sample_pixels = []
        
        for _ in range(n_samples):
            y = np.random.randint(h//4, 3*h//4)
            x = np.random.randint(w//4, 3*w//4)
            
            signals = self.corrected_stack[:, y, x]
            
            if signals[0] > self.noise_level * 3:  # Only consider pixels with good SNR
                # Check if decay is present
                decay_ratio = signals[-1] / signals[0] if signals[0] > 0 else 1
                sample_pixels.append({
                    'coords': (x, y),
                    'signals': signals,
                    'decay_ratio': decay_ratio
                })
        
        # Analyze decay patterns
        decay_ratios = [p['decay_ratio'] for p in sample_pixels]
        
        print(f"\nPixel decay analysis ({len(sample_pixels)} pixels sampled):")
        print(f"  Mean decay ratio (last/first): {np.mean(decay_ratios):.3f}")
        print(f"  Std dev: {np.std(decay_ratios):.3f}")
        print(f"  Min/Max: {np.min(decay_ratios):.3f} / {np.max(decay_ratios):.3f}")
        
        # Count problematic pixels
        no_decay = sum(1 for r in decay_ratios if r > 0.95)
        inverted = sum(1 for r in decay_ratios if r > 1.05)
        
        if no_decay > len(decay_ratios) * 0.5:
            print(f"\nWARNING: {no_decay}/{len(decay_ratios)} pixels show little/no decay!")
            print("Possible issues:")
            print("  - TE prep times may be too short")
            print("  - TE prep values may be incorrect")
            print("  - Images may not be T2-weighted")

    def is_septum_roi(self, roi_coords):
        """
        Detect if ROI is likely in the septum based on shape and location
        """
        x1, x2, y1, y2 = roi_coords
        roi_width = x2 - x1
        roi_height = y2 - y1
        aspect_ratio = roi_height / roi_width if roi_width > 0 else 1
        
        # Septum ROIs are typically tall and thin
        is_thin = roi_width < 30 or aspect_ratio > 2
        
        # Check if in center of image
        img_center_x = self.corrected_stack.shape[2] // 2
        roi_center_x = (x1 + x2) // 2
        is_central = abs(roi_center_x - img_center_x) < img_center_x * 0.3
        
        return is_thin and is_central

    def apply_septum_specific_corrections(self, t2_map, r2_map):
        """
        Apply specific corrections for septal measurements
        """
        # Median filter to reduce outliers
        from scipy.ndimage import median_filter
        t2_filtered = median_filter(t2_map, size=3)
        
        # Replace low confidence pixels with filtered values
        low_confidence = r2_map < 0.85
        t2_map[low_confidence] = t2_filtered[low_confidence]
        
        # Cap extreme values more aggressively for septum
        t2_map[t2_map > 70] = 70  # Septum rarely exceeds 70ms
        t2_map[t2_map < 35] = 35  # Septum rarely below 35ms
        
        return t2_map
    def estimate_regional_noise(self, image, roi_mask=None):
        """
        Estimate noise level regionally for better septal measurements
        """
        if roi_mask is None:
            # Use the whole image
            h, w = image.shape
            roi_mask = np.ones((h, w), dtype=bool)
        
        # Method 1: Wavelet-based noise estimation
        try:
            import pywt
            # Decompose image
            coeffs = pywt.dwt2(image, 'db1')
            # Use diagonal detail coefficients
            sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
            return sigma
        except:
            pass
        
        # Method 2: Local standard deviation in background
        from scipy.ndimage import generic_filter
        
        # Calculate local std in 5x5 windows
        local_std = generic_filter(image, np.std, size=5)
        
        # Find regions with low gradient (likely noise)
        grad_x = np.gradient(image, axis=0)
        grad_y = np.gradient(image, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold for background
        background_mask = (grad_mag < np.percentile(grad_mag, 20)) & roi_mask
        
        if np.sum(background_mask) > 100:
            # Robust estimate using MAD
            noise_std = np.median(local_std[background_mask])
            return noise_std * 1.4826  # MAD to std conversion
        else:
            # Fallback to global estimate
            return self.noise_level
    def fit_t2_with_corrections(self, te_preps, signal, apply_noise_correction=True,
                           apply_b1_correction=False, apply_stim_echo_correction=False,
                           apply_t1_correction=False):
        """
        Fit T2 with advanced noise floor correction for cardiac imaging
        Without sklearn dependency
        """
        
        # Check if signal is valid
        if np.all(signal == 0) or np.any(np.isnan(signal)) or np.all(signal == signal[0]):
            return 0.0, 0.0, {}
        
        # Advanced noise floor correction
        if apply_noise_correction and self.noise_level is not None:
            # More aggressive truncation for cardiac imaging
            noise_threshold = 2.5 * self.noise_level  # Increased from 2.0
            
            # Estimate SNR for each echo time
            snr_per_echo = signal / self.noise_level
            
            # Dynamic truncation based on SNR
            valid_idx = (signal > noise_threshold) & (snr_per_echo > 3)
            
            if np.sum(valid_idx) < 3:
                # Try less strict criteria
                valid_idx = signal > (1.5 * self.noise_level)
                if np.sum(valid_idx) < 2:
                    return 0.0, 0.0, {}
            
            te_valid = te_preps[valid_idx]
            signal_valid = signal[valid_idx]
            
            # Apply magnitude correction for Rician noise
            signal_corrected = np.zeros_like(signal_valid)
            for i, (s, snr) in enumerate(zip(signal_valid, snr_per_echo[valid_idx])):
                if snr > 3:
                    # Standard correction for high SNR
                    signal_corrected[i] = np.sqrt(max(s**2 - self.noise_level**2, 0))
                else:
                    # McGibney correction for low SNR
                    sigma = self.noise_level
                    signal_corrected[i] = np.sqrt(max(s**2 - 2*sigma**2, 0))
            
        else:
            signal_corrected = signal.copy()
            te_valid = te_preps
            valid_idx = np.ones(len(signal), dtype=bool)
            snr_per_echo = signal / (self.noise_level if self.noise_level else 1.0)
        
        # Choose fitting method based on data quality
        if self.fitting_method_var.get() == "3-parameter":
            # Advanced 3-parameter fitting with constraints
            def t2_model_3param(te, s0, t2, c):
                return s0 * np.exp(-te / t2) + c
            
            try:
                # Robust initial parameter estimation without sklearn
                if len(te_valid) >= 3:
                    # Estimate offset as the minimum signal value
                    offset_estimate = np.min(signal_corrected) * 0.8
                    signal_for_log = signal_corrected - offset_estimate
                    
                    # Only use points well above noise for initial estimate
                    high_snr_mask = signal_for_log > 2 * self.noise_level
                    if np.sum(high_snr_mask) >= 2:
                        log_signal = np.log(signal_for_log[high_snr_mask] + 1e-10)
                        te_for_log = te_valid[high_snr_mask]
                        
                        # Robust linear fit using iterative outlier rejection
                        # Instead of HuberRegressor, use custom robust fitting
                        slope, intercept = self.robust_linear_fit(te_for_log, log_signal)
                        
                        t2_init = -1.0 / slope if slope < 0 else 50.0
                        s0_init = np.exp(intercept) + offset_estimate
                    else:
                        t2_init = 50.0
                        s0_init = signal_corrected[0]
                        offset_estimate = 0
                else:
                    t2_init = 50.0
                    s0_init = signal_corrected[0]
                    offset_estimate = 0
                
                # Constrain offset to be non-negative and less than minimum signal
                c_init = max(0, min(offset_estimate, np.min(signal_corrected) * 0.5))
                
                # Weight by SNR
                weights = np.sqrt(snr_per_echo[valid_idx])
                weights = weights / np.max(weights)
                
                # Use scipy's curve_fit with sigma parameter for weighted fitting
                popt, pcov = opt.curve_fit(
                    t2_model_3param, te_valid, signal_corrected,
                    p0=[s0_init, t2_init, c_init],
                    sigma=1/weights,  # Inverse weights for curve_fit
                    bounds=([0, 20, 0], [np.inf, 100, np.min(signal_corrected)]),
                    maxfev=5000
                )
                
                s0, t2, c = popt
                
                # Additional constraint for septum
                # If T2 is unrealistically high, try 2-parameter fit
                if t2 > 70:
                    # Force 2-parameter fit
                    return self.fit_2param_with_fixed_offset(
                        te_valid, signal_corrected, self.noise_level
                    )
                
                # Calculate fit quality
                fitted = t2_model_3param(te_valid, s0, t2, c)
                residuals = (signal_corrected - fitted) * weights
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum(((signal_corrected - np.mean(signal_corrected)) * weights)**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
            except Exception as e:
                print(f"3-parameter fit failed: {e}")
                # Fallback to 2-parameter
                return self.fit_2param_with_fixed_offset(
                    te_valid, signal_corrected, self.noise_level
                )
        
        else:
            # 2-parameter model with fixed offset
            return self.fit_2param_with_fixed_offset(
                te_valid, signal_corrected, self.noise_level
            )
        
        # Quality check
        if r2 < 0.7 or t2 < 20 or t2 > 80:
            # Try simple truncated fit
            if len(te_valid) >= 3:
                t2_simple = self.simple_truncated_fit(te_valid[:3], signal_corrected[:3])
                if 30 < t2_simple < 70:
                    t2 = t2_simple
                    r2 = 0.5  # Lower confidence
        
        # Store parameters
        params = {
            's0': s0,
            't2': t2,
            'c': c,
            'r2': r2,
            'n_points_used': len(te_valid),
            'mean_snr': np.mean(snr_per_echo[valid_idx])
        }
        
        return t2, r2, params

    def robust_linear_fit(self, x, y, n_iter=3):
        """
        Robust linear regression using iterative outlier rejection
        Simple alternative to HuberRegressor
        """
        mask = np.ones(len(x), dtype=bool)
        
        for _ in range(n_iter):
            # Fit using current mask
            if np.sum(mask) < 2:
                break
                
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            
            # Calculate residuals
            predicted = slope * x + intercept
            residuals = np.abs(y - predicted)
            
            # Remove outliers (points with residuals > 2*median)
            median_residual = np.median(residuals[mask])
            if median_residual > 0:
                mask = residuals < (2.5 * median_residual)
        
        # Final fit
        if np.sum(mask) >= 2:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
        else:
            slope, intercept = np.polyfit(x, y, 1)
        
        return slope, intercept

    def fit_2param_with_fixed_offset(self, te_valid, signal_corrected, offset):
        """
        2-parameter fit with fixed offset
        """
        signal_minus_offset = signal_corrected - offset
        positive_mask = signal_minus_offset > 0
        
        if np.sum(positive_mask) < 2:
            return 0.0, 0.0, {}
        
        te_positive = te_valid[positive_mask]
        signal_positive = signal_minus_offset[positive_mask]
        
        # Log-linear fit
        log_signal = np.log(signal_positive)
        slope, intercept = np.polyfit(te_positive, log_signal, 1)
        
        t2 = -1.0 / slope if slope < 0 else 50.0
        t2 = np.clip(t2, 20, 80)  # Constrain to reasonable range
        
        s0 = np.exp(intercept)
        
        # Calculate R²
        fitted = s0 * np.exp(-te_valid / t2) + offset
        residuals = signal_corrected - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((signal_corrected - np.mean(signal_corrected))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        params = {
            's0': s0,
            't2': t2,
            'c': offset,
            'r2': r2,
            'n_points_used': len(te_positive)
        }
        
        return t2, r2, params

    def simple_truncated_fit(self, te, signal):
        """
        Simple exponential fit using only first few points
        """
        if len(te) < 2 or np.any(signal <= 0):
            return 50.0
        
        log_signal = np.log(signal)
        slope, _ = np.polyfit(te, log_signal, 1)
        t2 = -1.0 / slope if slope < 0 else 50.0
        
        return np.clip(t2, 30, 70)

    def estimate_noise_level(self):
        """Estimate noise level from background regions"""
        if self.image_stack is None:
            return
        
        # Use the LAST image (highest TE, lowest SNR)
        img = self.image_stack[-1]  # Changed from first to last
        
        # Find the mode of the histogram (likely background)
        hist, bins = np.histogram(img.flatten(), bins=256)
        mode_idx = np.argmax(hist[:len(hist)//4])  # Look in lower quarter
        mode_value = bins[mode_idx]
        
        # Estimate noise around the mode
        background_mask = img < mode_value * 2
        if np.sum(background_mask) > 100:
            # For Rician noise in magnitude images
            self.noise_level = np.std(img[background_mask])
        else:
            # More conservative fallback
            self.noise_level = np.percentile(img, 5)  # 5th percentile
        
        print(f"Estimated noise level: {self.noise_level:.2f}")

    def apply_motion_correction(self, image_stack):
        """Apply motion correction using SimpleITK"""
        corrected = []
        reference = sitk.GetImageFromArray(image_stack[0])
        
        print("\nApplying motion correction...")
        
        for i, img in enumerate(image_stack):
            if i == 0:
                corrected.append(img)
                continue
                
            moving = sitk.GetImageFromArray(img)
            
            # Registration
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=1.0, minStep=0.001, numberOfIterations=200)
            
            # Initialize
            initial_transform = sitk.CenteredTransformInitializer(
                reference, moving, sitk.Euler2DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
            registration_method.SetInitialTransform(initial_transform)
            
            # Multi-resolution
            registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
            
            try:
                final_transform = registration_method.Execute(reference, moving)
                resampled = sitk.Resample(moving, reference, final_transform,
                                        sitk.sitkLinear, 0.0, moving.GetPixelID())
                corrected.append(sitk.GetArrayFromImage(resampled))
                print(f"  Corrected image {i+1}")
            except Exception as e:
                print(f"  Motion correction failed for image {i+1}: {e}")
                corrected.append(img)
            
        return np.stack(corrected, axis=0)

    def show_image_overview(self):
        """Show overview of loaded images"""
        if self.corrected_stack is None:
            return
            
        self.fig.clear()
        
        n_images = len(self.te_preps)
        rows = int(np.ceil(np.sqrt(n_images)))
        cols = int(np.ceil(n_images / rows))
        
        for i in range(n_images):
            ax = self.fig.add_subplot(rows, cols, i+1)
            
            # Show image with consistent windowing
            vmin = np.percentile(self.corrected_stack[i], 1)
            vmax = np.percentile(self.corrected_stack[i], 99)
            
            ax.imshow(self.corrected_stack[i], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"TE prep = {self.te_preps[i]:.1f} ms")
            ax.axis('off')
            
        self.fig.suptitle("T2prep Image Series")
        self.fig.tight_layout()
        self.canvas.draw()

    def get_manual_te_preps(self, n_images):
        """Get T2prep times manually from user"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter T2prep Times")
        dialog.geometry("400x500")
        
        ttk.Label(dialog, text="Enter T2 preparation times (ms):", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        ttk.Label(dialog, text="Common T2prep sequences use values like:\n"
                            "0, 25, 50, 75 ms\n"
                            "0, 30, 60, 90 ms\n"
                            "0, 40, 80, 120 ms", 
                 font=("Arial", 10)).pack(pady=5)
        
        # Frame for entries
        entry_frame = ttk.Frame(dialog)
        entry_frame.pack(pady=10)
        
        # Create entry fields
        entries = []
        for i in range(n_images):
            frame = ttk.Frame(entry_frame)
            frame.pack(pady=5)
            ttk.Label(frame, text=f"Image {i+1}:", width=10).pack(side=tk.LEFT, padx=5)
            entry = ttk.Entry(frame, width=10)
            entry.pack(side=tk.LEFT)
            
            # Suggest default values
            if n_images == 4:
                defaults = [0, 25, 50, 75]
            elif n_images == 3:
                defaults = [0, 30, 60]
            else:
                defaults = [i * 25 for i in range(n_images)]
                
            if i < len(defaults):
                entry.insert(0, str(defaults[i]))
                
            entries.append(entry)
        
        result = {'values': None}
        
        def on_ok():
            try:
                values = [float(e.get()) for e in entries]
                # Check for reasonable values
                if any(v < 0 or v > 500 for v in values):
                    messagebox.showerror("Error", "TE prep times should be between 0 and 500 ms")
                    return
                if len(set(values)) < 2:
                    messagebox.showerror("Error", "At least 2 different TE prep times are required")
                    return
                result['values'] = values
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        def on_cancel():
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)
        
        return result['values']

    def calculate_t2(self):
        """Calculate T2 map with ROI selection"""
        if self.corrected_stack is None:
            messagebox.showinfo("Info", "Please load DICOM files first")
            return
            
        try:
            self.update_status("Selecting ROI...")
            
            # Show average image for ROI selection
            avg_image = np.mean(self.corrected_stack, axis=0)
            
            # ROI selection
            self.roi_coords = self.select_roi(avg_image)
            if not self.roi_coords:
                return
                
            x1, x2, y1, y2 = self.roi_coords
            roi_data = self.corrected_stack[:, y1:y2, x1:x2]
            
            self.update_status("Calculating T2 map...")
            
            # Progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("T2 Calculation")
            progress_window.geometry("400x150")
            
            ttk.Label(progress_window, text="Calculating T2 map...", 
                     font=("Arial", 12)).pack(pady=10)

            # In calculate_t2() method, after the pixel loop and before displaying results:

            # Apply spatial regularization for septum
            if self.is_thin_roi(self.roi_coords):
                print("Applying septum-specific post-processing...")
                
                # 1. Identify outliers using local statistics
                from scipy.ndimage import generic_filter
                
                # Calculate local median in 3x3 windows
                t2_median_filtered = median_filter(self.t2_map, size=3)
                
                # Find pixels that deviate significantly from local median
                deviation = np.abs(self.t2_map - t2_median_filtered)
                outlier_threshold = 15  # ms
                outliers = (deviation > outlier_threshold) & (self.t2_map > 0)
                
                # Replace outliers with median filtered values
                self.t2_map[outliers] = t2_median_filtered[outliers]
                
                # 2. Apply stronger constraints for septum
                # Normal septal T2 is typically 45-55ms at 1.5T, 50-60ms at 3T
                self.t2_map = np.clip(self.t2_map, 35, 70)
                
                # 3. Additional smoothing for low R² regions
                low_confidence = self.r2_map < 0.85
                if np.any(low_confidence):
                    # Apply Gaussian smoothing only to low confidence regions
                    t2_smooth = gaussian_filter(self.t2_map, sigma=1)
                    self.t2_map[low_confidence] = t2_smooth[low_confidence]
                
                print(f"Septum post-processing: corrected {np.sum(outliers)} outlier pixels")

            def is_thin_roi(self, roi_coords):
                """Check if ROI is thin (likely septum)"""
                x1, x2, y1, y2 = roi_coords
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width if width > 0 else 1
                
                return width < 40 or aspect_ratio > 1.5
            progress = ttk.Progressbar(progress_window, length=300, mode="determinate")
            progress.pack(pady=10)
            
            # Calculate T2 for each pixel
            roi_height, roi_width = roi_data.shape[1:]
            self.t2_map = np.zeros((roi_height, roi_width))
            self.r2_map = np.zeros((roi_height, roi_width))
            progress["maximum"] = roi_height * roi_width
            
            pixel_count = 0
            for y in range(roi_height):
                for x in range(roi_width):
                    pixel_signal = roi_data[:, y, x]
                    
                    # Fit T2 with corrections
                    t2, r2, fitted_params = self.fit_t2_with_corrections(
                        self.te_preps, pixel_signal,
                        apply_noise_correction=self.noise_correction_var.get(),
                        apply_b1_correction=self.b1_correction_var.get(),
                        apply_stim_echo_correction=self.stim_echo_var.get(),
                        apply_t1_correction=self.t1_correction_var.get()
                    )
                    
                    self.t2_map[y, x] = t2
                    self.r2_map[y, x] = r2
                    
                    # Cache pixel data
                    global_x = x1 + x
                    global_y = y1 + y
                    self.pixel_cache[(global_x, global_y)] = {
                        'signal': pixel_signal,
                        't2': t2,
                        'r2': r2,
                        'params': fitted_params
                    }
                    
                    pixel_count += 1
                    progress["value"] = pixel_count
                    progress_window.update()
                    
            progress_window.destroy()

            # Post-process the T2 map to remove artifacts and improve quality
            from scipy.ndimage import median_filter

            # Remove outliers using median filter
            t2_median = median_filter(self.t2_map, size=3)
            outlier_mask = np.abs(self.t2_map - t2_median) > 50
            self.t2_map[outlier_mask] = t2_median[outlier_mask]

            # Set physiologically implausible values to zero
            self.t2_map[(self.t2_map < 20) | (self.t2_map > 150)] = 0

            # Optional: Also clean up the R² map
            self.r2_map[self.t2_map == 0] = 0

            print(f"Post-processing complete: removed {np.sum(outlier_mask)} outlier pixels")

            
            # Calculate statistics
            valid_mask = (self.t2_map > 15) & (self.t2_map < 120) & (self.r2_map > 0.85)
            if np.any(valid_mask):
                valid_t2 = self.t2_map[valid_mask]
                mean_t2 = np.mean(valid_t2)
                median_t2 = np.median(valid_t2)
                std_t2 = np.std(valid_t2)
                
                self.t2_result_var.set(f"Mean T2: {mean_t2:.1f} ms\n"
                                     f"Median T2: {median_t2:.1f} ms\n"
                                     f"Std Dev: {std_t2:.1f} ms\n"
                                     f"Valid pixels: {np.sum(valid_mask)}")
                
                # Auto-set display range
                self.t2_min_var.set("0")
                self.t2_max_var.set(f"{np.percentile(valid_t2, 95):.0f}")
            
            # Enable save buttons
            self.save_button.configure(state=tk.NORMAL)
            
            # Display T2 map
            self.display_t2_map()
            
            self.update_status("T2 calculation complete")

            n_failed = np.sum((self.r2_map < 0.8) | (self.t2_map == 0))
            success_rate = 100 * (1 - n_failed / self.t2_map.size)
            print(f"Fitting success rate: {success_rate:.1f}%")
            
        except Exception as e:
            self.update_status("Error calculating T2")
            messagebox.showerror("Error", f"Error calculating T2: {str(e)}")

    def calculate_full_t2_map(self):
        """Calculate T2 map for entire image"""
        if self.corrected_stack is None:
            messagebox.showinfo("Info", "Please load DICOM files first")
            return
            
        # Set full image as ROI
        height, width = self.corrected_stack.shape[1:]
        self.roi_coords = (0, width, 0, height)
        
        # Calculate using existing method
        self.calculate_t2()
        
        # Store as full map
        self.t2_map_full = self.t2_map.copy()

    def select_roi(self, image):
        """Select ROI using matplotlib"""
        coords = {'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0}
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap='gray')
        ax.set_title("Click and drag to select ROI\nClose window when done")
        
        def onselect(eclick, erelease):
            coords['x1'] = int(min(eclick.xdata, erelease.xdata))
            coords['x2'] = int(max(eclick.xdata, erelease.xdata))
            coords['y1'] = int(min(eclick.ydata, erelease.ydata))
            coords['y2'] = int(max(eclick.ydata, erelease.ydata))
            
        rect_selector = RectangleSelector(ax, onselect, interactive=True,
                                        minspanx=5, minspany=5)
        
        plt.show()
        
        if coords['x2'] > coords['x1'] and coords['y2'] > coords['y1']:
            return coords['x1'], coords['x2'], coords['y1'], coords['y2']
        return None

    def display_t2_map(self):
        """Display T2 map with interactive features"""
        if self.t2_map is None:
            return
            
        self.fig.clear()
        
        # Create layout
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
        
        # Main T2 map
        self.main_ax = self.fig.add_subplot(gs[:, 0])
        
        # Get display parameters
        cmap = self.colormap_var.get()
        vmin = float(self.t2_min_var.get())
        vmax = float(self.t2_max_var.get())
        
        # Show ROI in context
        if hasattr(self, 'roi_coords') and self.roi_coords:
            x1, x2, y1, y2 = self.roi_coords
            # Show background
            bg_image = np.mean(self.corrected_stack, axis=0)
            self.main_ax.imshow(bg_image, cmap='gray', alpha=0.5)
            
            # Overlay T2 map
            extent = [x1, x2, y1, y2]
            im = self.main_ax.imshow(self.t2_map, cmap=cmap, vmin=vmin, vmax=vmax,
                                   extent=extent, alpha=0.9)
            
            # ROI box
            from matplotlib.patches import Rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                           edgecolor='red', linewidth=2)
            self.main_ax.add_patch(rect)
        else:
            im = self.main_ax.imshow(self.t2_map, cmap=cmap, vmin=vmin, vmax=vmax)
        
        self.main_ax.set_title("T2 Map")
        self.main_ax.axis('off')
        
        # Colorbar
        cbar = self.fig.colorbar(im, ax=self.main_ax)
        cbar.set_label('T2 (ms)')
        
        # Fit display
        self.fit_ax = self.fig.add_subplot(gs[0, 1])
        self.fit_ax.set_title("T2 Decay Fit")
        self.fit_ax.set_xlabel("TE prep (ms)")
        self.fit_ax.set_ylabel("Signal")
        self.fit_ax.grid(True, alpha=0.3)
        
        # R² map
        self.r2_ax = self.fig.add_subplot(gs[1, 1])
        if self.r2_map is not None:
            im_r2 = self.r2_ax.imshow(self.r2_map, cmap='hot', vmin=0, vmax=1)
            self.r2_ax.set_title("R² Map (Fit Quality)")
            self.r2_ax.axis('off')
            cbar_r2 = self.fig.colorbar(im_r2, ax=self.r2_ax)
            cbar_r2.set_label('R²')
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Handle mouse movement for pixel inspection"""
        if event.inaxes != self.main_ax or self.t2_map is None:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        if hasattr(self, 'roi_coords') and self.roi_coords:
            x1, x2, y1, y2 = self.roi_coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                if (x, y) in self.pixel_cache:
                    self.update_pixel_display(x, y)

    def update_pixel_display(self, x, y):
        """Update fit display for current pixel"""
        if (x, y) not in self.pixel_cache:
            return
            
        data = self.pixel_cache[(x, y)]
        signal = data['signal']
        t2_value = data['t2']
        r2_value = data['r2']
        params = data.get('params', {})
        
        self.fit_ax.clear()
        
        # Plot data
        self.fit_ax.scatter(self.te_preps, signal, color='blue', s=50, 
                          label='Measured', zorder=5)
        
        # Plot fit
        if t2_value > 0:
            te_smooth = np.linspace(0, max(self.te_preps), 100)
            
            if self.fitting_method_var.get() == "3-parameter":
                s0 = params.get('s0', signal[0])
                c = params.get('c', 0)
                fit_signal = s0 * np.exp(-te_smooth / t2_value) + c
            else:
                s0 = params.get('s0', signal[0])
                fit_signal = s0 * np.exp(-te_smooth / t2_value)
            
            # Get confidence interval if available
            t2_ci = params.get('t2_ci', 0)
            if t2_ci > 0:
                label_text = f'T2 = {t2_value:.1f} ± {t2_ci:.1f} ms\nR² = {r2_value:.3f}'
            else:
                label_text = f'T2 = {t2_value:.1f} ms\nR² = {r2_value:.3f}'
                
            self.fit_ax.plot(te_smooth, fit_signal, 'r-', 
                           label=label_text,
                           linewidth=2)
        
        # Show noise floor
        if self.noise_correction_var.get() and self.noise_level:
            self.fit_ax.axhline(y=self.noise_level, color='gray', 
                              linestyle='--', alpha=0.5, 
                              label=f'Noise = {self.noise_level:.1f}')
        
        self.fit_ax.set_xlabel("TE prep (ms)")
        self.fit_ax.set_ylabel("Signal")
        self.fit_ax.legend()
        self.fit_ax.grid(True, alpha=0.3)
        
        # Title with coordinates
        roi_x = x - self.roi_coords[0]
        roi_y = y - self.roi_coords[1]
        self.fit_ax.set_title(f"Pixel ({x}, {y}) - ROI ({roi_x}, {roi_y})")
        
        self.canvas.draw_idle()

    def update_display(self, event=None):
        """Update the display"""
        if self.t2_map is not None:
            self.display_t2_map()

    def save_t2_map(self):
        """Save T2 map"""
        if self.t2_map is None:
            messagebox.showinfo("Info", "No T2 map to save")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save T2 Map",
                defaultextension=".npy",
                filetypes=[("NumPy files", "*.npy"), ("PNG files", "*.png"),
                          ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
                
            if file_path.endswith('.npy'):
                # Save with metadata
                save_data = {
                    't2_map': self.t2_map,
                    'r2_map': self.r2_map,
                    'te_preps': self.te_preps,
                    'roi_coords': self.roi_coords,
                    'noise_level': self.noise_level,
                    'corrections_applied': {
                        'noise': self.noise_correction_var.get(),
                        'b1': self.b1_correction_var.get(),
                        'stim_echo': self.stim_echo_var.get(),
                        't1': self.t1_correction_var.get()
                    }
                }
                np.save(file_path, save_data)
                
            elif file_path.endswith('.png'):
                plt.imsave(file_path, self.t2_map, cmap=self.colormap_var.get(),
                          vmin=float(self.t2_min_var.get()),
                          vmax=float(self.t2_max_var.get()))
                
            elif file_path.endswith('.csv'):
                np.savetxt(file_path, self.t2_map, delimiter=',', fmt='%.2f')
                
            self.update_status(f"T2 map saved to {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving T2 map: {str(e)}")


# Main entry point
def main():
    """Main entry point"""
    root = tk.Tk()
    app = T2MappingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
