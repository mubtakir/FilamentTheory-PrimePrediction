#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مستكشف الأعداد الأولية التفاعلي
Interactive Prime Explorer

واجهة مستخدم تفاعلية لاستكشاف أفكار الباحث العلمي باسل يحيى عبدالله
Interactive user interface for exploring researcher Basel Yahya Abdullah's ideas

الباحث العلمي: باسل يحيى عبدالله (Basel Yahya Abdullah)
المطور: مبتكر (Mubtakir)
التاريخ: 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from hpp_predictor import HybridPrimePredictor
from advanced_prime_algorithm import AdvancedPrimeFinder
from mathematical_analysis import MathematicalPrimeAnalysis


class PrimeExplorerGUI:
    """واجهة المستخدم الرسومية لمستكشف الأعداد الأولية"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("مستكشف الأعداد الأولية المتقدم - Advanced Prime Explorer")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # تهيئة المكونات
        self.predictor = HybridPrimePredictor()
        self.finder = AdvancedPrimeFinder()
        self.analyzer = MathematicalPrimeAnalysis()
        
        # متغيرات الواجهة
        self.is_running = False
        
        self.setup_ui()
        self.load_models()
    
    def setup_ui(self):
        """إعداد واجهة المستخدم"""
        # العنوان الرئيسي
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="مستكشف الأعداد الأولية المتقدم\nAdvanced Prime Explorer",
                              font=('Arial', 16, 'bold'),
                              fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # إطار الأدوات
        tools_frame = tk.Frame(self.root, bg='#ecf0f1')
        tools_frame.pack(fill='x', padx=5, pady=5)
        
        # الأدوات - الصف الأول
        row1_frame = tk.Frame(tools_frame, bg='#ecf0f1')
        row1_frame.pack(fill='x', pady=2)
        
        # إيجاد العدد الأولي التالي
        tk.Label(row1_frame, text="العدد:", bg='#ecf0f1').pack(side='left', padx=5)
        self.number_entry = tk.Entry(row1_frame, width=15)
        self.number_entry.pack(side='left', padx=5)
        
        tk.Button(row1_frame, text="العدد الأولي التالي", 
                 command=self.find_next_prime,
                 bg='#3498db', fg='white').pack(side='left', padx=5)
        
        tk.Button(row1_frame, text="التنبؤ الديناميكي", 
                 command=self.dynamic_prediction,
                 bg='#e74c3c', fg='white').pack(side='left', padx=5)
        
        # الأدوات - الصف الثاني
        row2_frame = tk.Frame(tools_frame, bg='#ecf0f1')
        row2_frame.pack(fill='x', pady=2)
        
        tk.Label(row2_frame, text="النطاق:", bg='#ecf0f1').pack(side='left', padx=5)
        self.start_entry = tk.Entry(row2_frame, width=10)
        self.start_entry.pack(side='left', padx=2)
        
        tk.Label(row2_frame, text="إلى", bg='#ecf0f1').pack(side='left', padx=2)
        self.end_entry = tk.Entry(row2_frame, width=10)
        self.end_entry.pack(side='left', padx=2)
        
        tk.Button(row2_frame, text="الغربال المقطعي", 
                 command=self.segmented_sieve,
                 bg='#27ae60', fg='white').pack(side='left', padx=5)
        
        tk.Button(row2_frame, text="تحليل رياضي", 
                 command=self.mathematical_analysis,
                 bg='#f39c12', fg='white').pack(side='left', padx=5)
        
        # الأدوات - الصف الثالث
        row3_frame = tk.Frame(tools_frame, bg='#ecf0f1')
        row3_frame.pack(fill='x', pady=2)
        
        tk.Button(row3_frame, text="مصفوفة هيلبرت-بوليا", 
                 command=self.hilbert_polya_matrix,
                 bg='#9b59b6', fg='white').pack(side='left', padx=5)
        
        tk.Button(row3_frame, text="اختبار جولدباخ", 
                 command=self.goldbach_test,
                 bg='#1abc9c', fg='white').pack(side='left', padx=5)
        
        tk.Button(row3_frame, text="الأعداد التوأم", 
                 command=self.twin_primes,
                 bg='#e67e22', fg='white').pack(side='left', padx=5)
        
        tk.Button(row3_frame, text="مسح النتائج", 
                 command=self.clear_results,
                 bg='#95a5a6', fg='white').pack(side='right', padx=5)
        
        # شريط التقدم
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill='x', padx=5, pady=2)
        
        # منطقة النتائج
        results_frame = tk.Frame(self.root)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        tk.Label(results_frame, text="النتائج:", font=('Arial', 12, 'bold')).pack(anchor='w')
        
        self.results_text = scrolledtext.ScrolledText(results_frame, 
                                                     font=('Courier', 10),
                                                     wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True)
        
        # شريط الحالة
        self.status_var = tk.StringVar()
        self.status_var.set("جاهز - Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor='w', bg='#bdc3c7')
        status_bar.pack(fill='x', side='bottom')
    
    def load_models(self):
        """تحميل النماذج المدربة"""
        try:
            self.predictor.load_models()
            self.log_result("✓ تم تحميل النماذج بنجاح")
        except Exception as e:
            self.log_result(f"⚠ تحذير: لم يتم تحميل النماذج - {str(e)}")
    
    def log_result(self, message):
        """إضافة رسالة إلى منطقة النتائج"""
        timestamp = time.strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def set_status(self, message):
        """تحديث شريط الحالة"""
        self.status_var.set(message)
        self.root.update()
    
    def start_progress(self):
        """بدء شريط التقدم"""
        self.progress.start(10)
        self.is_running = True
    
    def stop_progress(self):
        """إيقاف شريط التقدم"""
        self.progress.stop()
        self.is_running = False
    
    def find_next_prime(self):
        """إيجاد العدد الأولي التالي"""
        try:
            number = int(self.number_entry.get())
            self.set_status("البحث عن العدد الأولي التالي...")
            self.start_progress()
            
            def worker():
                try:
                    next_prime = self.finder.find_next_prime_optimized(number)
                    self.log_result(f"العدد الأولي التالي بعد {number} هو: {next_prime}")
                except Exception as e:
                    self.log_result(f"خطأ: {str(e)}")
                finally:
                    self.stop_progress()
                    self.set_status("جاهز")
            
            threading.Thread(target=worker, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("خطأ", "يرجى إدخال رقم صحيح")
    
    def dynamic_prediction(self):
        """التنبؤ الديناميكي"""
        try:
            number = int(self.number_entry.get())
            self.set_status("التنبؤ الديناميكي...")
            self.start_progress()
            
            def worker():
                try:
                    next_prime = self.predictor.predict_next_prime_dynamic(number)
                    if next_prime:
                        self.log_result(f"التنبؤ الديناميكي: العدد الأولي التالي بعد {number} هو {next_prime}")
                    else:
                        self.log_result(f"فشل التنبؤ الديناميكي للعدد {number}")
                except Exception as e:
                    self.log_result(f"خطأ في التنبؤ الديناميكي: {str(e)}")
                finally:
                    self.stop_progress()
                    self.set_status("جاهز")
            
            threading.Thread(target=worker, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("خطأ", "يرجى إدخال رقم صحيح")
    
    def segmented_sieve(self):
        """الغربال المقطعي"""
        try:
            start = int(self.start_entry.get())
            end = int(self.end_entry.get())
            
            if start >= end:
                messagebox.showerror("خطأ", "يجب أن يكون البداية أصغر من النهاية")
                return
            
            self.set_status("تشغيل الغربال المقطعي...")
            self.start_progress()
            
            def worker():
                try:
                    primes = self.finder.advanced_segmented_sieve(start, end)
                    self.log_result(f"الغربال المقطعي [{start}, {end}]:")
                    self.log_result(f"عدد الأعداد الأولية: {len(primes)}")
                    if len(primes) <= 50:
                        self.log_result(f"الأعداد الأولية: {primes}")
                    else:
                        self.log_result(f"أول 25: {primes[:25]}")
                        self.log_result(f"آخر 25: {primes[-25:]}")
                except Exception as e:
                    self.log_result(f"خطأ في الغربال المقطعي: {str(e)}")
                finally:
                    self.stop_progress()
                    self.set_status("جاهز")
            
            threading.Thread(target=worker, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("خطأ", "يرجى إدخال أرقام صحيحة")
    
    def mathematical_analysis(self):
        """التحليل الرياضي"""
        self.set_status("تشغيل التحليل الرياضي...")
        self.start_progress()
        
        def worker():
            try:
                self.log_result("=== التحليل الرياضي للأعداد الأولية ===")
                
                # تحليل نظرية الأعداد الأولية
                x_values = [100, 500, 1000]
                pnt_results = self.analyzer.prime_number_theorem_analysis(x_values)
                
                self.log_result("تحليل نظرية الأعداد الأولية:")
                for i, x in enumerate(x_values):
                    actual = pnt_results['actual_pi_x'][i]
                    estimate = pnt_results['pnt_estimate'][i]
                    error = pnt_results['errors'][i]
                    self.log_result(f"π({x}) = {actual}, تقدير = {estimate:.1f}, خطأ = {error:.2f}%")
                
            except Exception as e:
                self.log_result(f"خطأ في التحليل الرياضي: {str(e)}")
            finally:
                self.stop_progress()
                self.set_status("جاهز")
        
        threading.Thread(target=worker, daemon=True).start()
    
    def hilbert_polya_matrix(self):
        """مصفوفة هيلبرت-بوليا"""
        self.set_status("بناء مصفوفة هيلبرت-بوليا...")
        self.start_progress()
        
        def worker():
            try:
                matrix, eigenvalues = self.predictor.experimental_hilbert_polya_matrix(30)
                self.log_result("=== مصفوفة هيلبرت-بوليا التجريبية ===")
                self.log_result(f"حجم المصفوفة: {matrix.shape}")
                self.log_result(f"عدد القيم الذاتية: {len(eigenvalues)}")
                self.log_result(f"أكبر 5 قيم ذاتية: {eigenvalues[-5:]}")
                
                # مقارنة مع أصفار زيتا
                self.predictor.compare_with_zeta_zeros(eigenvalues, 5)
                
            except Exception as e:
                self.log_result(f"خطأ في مصفوفة هيلبرت-بوليا: {str(e)}")
            finally:
                self.stop_progress()
                self.set_status("جاهز")
        
        threading.Thread(target=worker, daemon=True).start()
    
    def goldbach_test(self):
        """اختبار حدسية جولدباخ"""
        self.set_status("اختبار حدسية جولدباخ...")
        self.start_progress()
        
        def worker():
            try:
                results = self.analyzer.goldbach_conjecture_test(100)
                self.log_result("=== اختبار حدسية جولدباخ ===")
                self.log_result(f"الحالات المتحققة: {results['verified_cases']}")
                self.log_result(f"معدل النجاح: {results['success_rate']:.2f}%")
                self.log_result(f"أمثلة: {results['sample_decompositions'][:5]}")
                
            except Exception as e:
                self.log_result(f"خطأ في اختبار جولدباخ: {str(e)}")
            finally:
                self.stop_progress()
                self.set_status("جاهز")
        
        threading.Thread(target=worker, daemon=True).start()
    
    def twin_primes(self):
        """الأعداد الأولية التوأم"""
        self.set_status("البحث عن الأعداد الأولية التوأم...")
        self.start_progress()
        
        def worker():
            try:
                results = self.analyzer.twin_primes_analysis(1000)
                self.log_result("=== الأعداد الأولية التوأم ===")
                self.log_result(f"عدد الأزواج التوأم: {results['count']}")
                self.log_result(f"الكثافة: {results['density']:.4f}%")
                self.log_result(f"أول 5 أزواج: {results['first_10'][:5]}")
                
            except Exception as e:
                self.log_result(f"خطأ في الأعداد التوأم: {str(e)}")
            finally:
                self.stop_progress()
                self.set_status("جاهز")
        
        threading.Thread(target=worker, daemon=True).start()
    
    def clear_results(self):
        """مسح النتائج"""
        self.results_text.delete(1.0, tk.END)
        self.set_status("تم مسح النتائج")


def main():
    """تشغيل التطبيق"""
    root = tk.Tk()
    app = PrimeExplorerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
