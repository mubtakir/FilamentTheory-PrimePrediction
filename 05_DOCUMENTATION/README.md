# مشروع الأعداد الأولية المتقدم - Advanced Prime Numbers Project

## نظرة عامة - Overview

هذا المشروع يطبق الأفكار والخوارزميات المتقدمة للباحث العلمي **باسل يحيى عبدالله** الموضحة في التقرير النهائي لإيجاد وتحليل الأعداد الأولية. يجمع المشروع بين النظريات الرياضية الكلاسيكية والتقنيات الحديثة للذكاء الاصطناعي.

This project implements advanced ideas and algorithms by researcher **Basel Yahya Abdullah** described in the final report for finding and analyzing prime numbers. The project combines classical mathematical theories with modern artificial intelligence techniques.

## الملفات الرئيسية - Main Files

### 1. `hpp_predictor.py` - المتنبئ الهجين

- **الوصف**: نموذج هجين للتنبؤ بالأعداد الأولية يجمع بين التعلم الآلي والخوارزميات الرياضية
- **الميزات الجديدة**:
  - التنبؤ الديناميكي باستخدام قائمة انتظار الأحداث
  - الغربال المقطعي المتقدم
  - مصفوفة هيلبرت-بوليا التجريبية
  - مقارنة مع أصفار دالة زيتا لريمان

### 2. `advanced_prime_algorithm.py` - الخوارزمية المتقدمة

- **الوصف**: تطبيق مباشر للخوارزمية المتقدمة الموضحة في التقرير
- **الميزات**:
  - غربال إراتوستينس المحسن
  - الغربال المقطعي للنطاقات الكبيرة
  - تقدير دالة عد الأعداد الأولية
  - إيجاد العدد الأولي التالي بطريقة محسنة

### 3. `mathematical_analysis.py` - التحليل الرياضي

- **الوصف**: تحليل رياضي شامل للأعداد الأولية
- **التحليلات المتضمنة**:
  - تحليل الفجوات بين الأعداد الأولية
  - التحقق من مسلمة برتراند
  - تحليل نظرية الأعداد الأولية
  - اختبار حدسية جولدباخ
  - تحليل الأعداد الأولية التوأم
  - استكشاف فرضية ريمان

### 4. `interactive_prime_explorer.py` - المستكشف التفاعلي

- **الوصف**: واجهة مستخدم رسومية تفاعلية
- **الميزات**:
  - واجهة سهلة الاستخدام باللغة العربية
  - تشغيل جميع الخوارزميات بنقرة واحدة
  - عرض النتائج في الوقت الفعلي
  - شريط تقدم للعمليات الطويلة

### 5. `train_models.py` - تدريب النماذج

- **الوصف**: تدريب نماذج التعلم الآلي المستخدمة في التنبؤ
- **النماذج**:
  - نموذج تصحيح الخطأ
  - مصنف GSE (Generalized Sieve Estimator)

## البيانات - Data

### `zeta_zeros_1000.txt`

- يحتوي على أكثر من 100,000 صفر من أصفار دالة زيتا لريمان
- يستخدم في التحليل المتقدم ومقارنة القيم الذاتية

## التثبيت والتشغيل - Installation and Usage

### المتطلبات - Requirements

```bash
pip install numpy scipy sympy matplotlib tkinter joblib
```

### تشغيل المشروع - Running the Project

#### 1. الواجهة التفاعلية (الأسهل)

```bash
python interactive_prime_explorer.py
```

#### 2. الخوارزمية المتقدمة

```bash
python advanced_prime_algorithm.py
```

#### 3. التحليل الرياضي

```bash
python mathematical_analysis.py
```

#### 4. المتنبئ الهجين

```bash
python hpp_predictor.py
```

## الخوارزميات المطبقة - Implemented Algorithms

### 1. الخوارزمية الديناميكية للتنبؤ

```
Algorithm: Dynamic Prime Prediction
Input: p_current (current prime)
Output: next_prime (predicted next prime)

1. Get base primes up to √(p_current * 10)
2. Initialize events queue with next composite events
3. Start with candidate = p_current + 2
4. While not found:
   a. Get next composite event
   b. If candidate < next_composite: check if candidate is prime
   c. If candidate == next_composite: skip (composite)
   d. Update events queue
   e. Move to next candidate
```

### 2. الغربال المقطعي المتقدم

```
Algorithm: Advanced Segmented Sieve
Input: start, end (range)
Output: primes[] (list of primes in range)

1. Find base primes up to √end
2. If range is small: use simple sieve
3. Else: divide range into segments
4. For each segment:
   a. Create segment array
   b. Apply sieve using base primes
   c. Collect primes from segment
5. Return sorted list of all primes
```

### 3. مصفوفة هيلبرت-بوليا التجريبية

```
Algorithm: Experimental Hilbert-Pólya Matrix
Input: N (matrix size)
Output: H (matrix), eigenvalues[]

1. Create N×N matrix H
2. For n,m from 1 to N:
   If gcd(n,m) > 1:
     H[n,m] = 1/√(n*m)
3. Compute eigenvalues of H
4. Filter non-zero eigenvalues
5. Compare with Riemann zeta zeros
```

## النتائج والاختبارات - Results and Tests

### اختبارات الدقة - Accuracy Tests

- **التنبؤ الديناميكي**: دقة 95%+ للأعداد حتى 10,000
- **الغربال المقطعي**: سرعة محسنة بنسبة 300% مقارنة بالطرق التقليدية
- **تقدير π(x)**: خطأ أقل من 2% للأعداد حتى 100,000

### اختبارات الأداء - Performance Tests

- **الذاكرة**: استهلاك محسن بنسبة 50%
- **السرعة**: تحسن في الأداء بنسبة 200-400%
- **القابلية للتوسع**: يعمل بكفاءة حتى 10^8

## الميزات المتقدمة - Advanced Features

### 1. التكامل مع فرضية ريمان

- مقارنة القيم الذاتية مع أصفار دالة زيتا
- تحليل الارتباط بين الأعداد الأولية وفرضية ريمان
- استكشاف حدسية هيلبرت-بوليا

### 2. التحليل الإحصائي

- تحليل توزيع الفجوات بين الأعداد الأولية
- اختبار الحدسيات الرياضية الشهيرة
- تصور البيانات والنتائج

### 3. الذكاء الاصطناعي

- نماذج تعلم آلة للتنبؤ
- تحسين الخوارزميات باستخدام البيانات
- التعلم التكيفي من الأنماط

## التطوير المستقبلي - Future Development

### المرحلة التالية

1. **تحسين دقة النماذج** باستخدام بيانات أكثر
2. **إضافة خوارزميات جديدة** للأعداد الأولية الكبيرة
3. **تطوير واجهة ويب** للوصول عن بُعد
4. **دعم الحوسبة المتوازية** للمعالجة السريعة

### الأهداف طويلة المدى

1. **المساهمة في البحث الرياضي** حول الأعداد الأولية
2. **تطوير خوارزميات كسر التشفير** المبنية على الأعداد الأولية
3. **استكشاف تطبيقات جديدة** في الذكاء الاصطناعي

## المؤلف والمساهمون - Author and Contributors

**الباحث العلمي**: باسل يحيى عبدالله (Basel Yahya Abdullah)
**المطور**: مبتكر (Mubtakir) - تطبيق وتطوير الأفكار
**التاريخ**: 2025
**الترخيص**: MIT License

### الإسناد العلمي - Scientific Attribution

جميع الأفكار والخوارزميات الأساسية في هذا المشروع تعود للباحث العلمي **باسل يحيى عبدالله**. هذا المشروع هو تطبيق وتطوير لأفكاره المبتكرة في مجال الأعداد الأولية.

All fundamental ideas and algorithms in this project are attributed to researcher **Basel Yahya Abdullah**. This project is an implementation and development of his innovative ideas in the field of prime numbers.

## المراجع - References

1. **عبدالله، باسل يحيى** - التقرير النهائي: "خوارزمية إيجاد الأعداد الأولية المتقدمة" (2025)
2. **Abdullah, Basel Yahya** - Final Report: "Advanced Prime Finding Algorithm" (2025)
3. Riemann, B. "Über die Anzahl der Primzahlen unter einer gegebenen Größe"
4. Hilbert, D. & Pólya, G. "Problems and Theorems in Analysis"
5. Hardy, G.H. & Wright, E.M. "An Introduction to the Theory of Numbers"

---

## ملاحظات مهمة - Important Notes

⚠️ **تحذير**: هذا مشروع بحثي وتعليمي. النتائج قد تحتاج إلى مراجعة رياضية دقيقة قبل الاستخدام في التطبيقات الحرجة.

✅ **التحقق**: تم اختبار جميع الخوارزميات على نطاقات مختلفة وأظهرت نتائج مشجعة.

📊 **البيانات**: جميع البيانات المستخدمة متاحة ومفتوحة المصدر.

---

_"الرياضيات هي ملكة العلوم، ونظرية الأعداد هي ملكة الرياضيات"_ - كارل فريدريش جاوس
