# Forex & Crypto AI — موقع أونلاين جاهز للنشر

## ما الذي ستجده هنا؟
- `app.py` — موقع Gradio لتدريب LSTM، جلب بيانات فوركس/كريبتو، باكتيست، قرار اليوم، وإرسال POST.
- `requirements.txt` — الحزم.
- `render.yaml` — ملف نشر تلقائي على Render.

## نشر سريع بدون تعقيد
1) GitHub → أنشئ مستودع باسم `forex-crypto-ai` → ارفع الثلاثة ملفات كما هي.
2) Render.com → New → Web Service → اختر المستودع → سيعمل البناء والتشغيل تلقائيًا.
3) خذ الرابط النهائي، وبإمكانك إضافة دومينك من Custom Domains.

### إعدادات اختيارية (Environment)
- `SERVER_ENDPOINT` : لو عندك API خاص تستقبل منه القرارات (JSON).
- `ALPHA_VANTAGE_KEY` : مفتاح AlphaVantage لتحسين جلب بيانات الفوركس (وإلا يستخدم Yahoo).

## تشغيل محليًا (اختياري)
```
pip install -r requirements.txt
python app.py
```
ثم افتح المتصفح على: http://127.0.0.1:7860
