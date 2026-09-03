import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(
    page_title="EcoSort - AI Waste Classifier & Rewards",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white; font-weight: bold; border-radius: 12px;
        padding: 0.6rem 1.5rem; border: none; width: 100%;
        box-shadow: 0 4px 6px rgba(40, 167, 69, 0.2);
    }
    .prediction-card {
        padding: 15px; border-radius: 12px; background-color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center;
        border-left: 6px solid #28a745; margin-bottom: 15px;
    }
    .rank-card {
        padding: 12px; border-radius: 10px; background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #81c784; text-align: center; margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "users_db.json"

def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_database(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=4)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classifier_model.h5")
    return model

with st.spinner("🔄 جاري تحميل ذكاء EcoSort الاصطناعي..."):
    model = load_model()

class_names = ['Battery', 'Cardboard', 'Clothes', 'Glass', 'Metal', 'Paper', 'Plastic']
points_mapping = {'Battery': 50, 'Cardboard': 20, 'Clothes': 10, 'Glass': 30, 'Metal': 30, 'Paper': 10, 'Plastic': 20}
co2_mapping = {'Battery': 150.0, 'Cardboard': 40.0, 'Clothes': 80.0, 'Glass': 25.0, 'Metal': 200.0, 'Paper': 30.0, 'Plastic': 50.0}

recycling_tips = {
    'Battery': '⚠️ بطاريات خطرة! يجب التخلص منها في حاويات إعادة التدوير المخصصة للإلكترونيات.',
    'Cardboard': '📦 قم بطي صناديق الكرتون وتجفيفها جيداً قبل وضعها في سلة الورق.',
    'Clothes': '👕 الملابس السليمة يمكن التبرع بها، والمستهلكة توضع في مراكز إعادة تدوير الأقمشة.',
    'Glass': '🍾 الزجاج قابل لإعادة التدوير بنسبة 100% بدون فقدان في الجودة!',
    'Metal': '🥫 العلب المعدنية والألمنيوم توفر طاقة هائلة عند إعادة تدويرها.',
    'Paper': '📄 حافظ على نظافة الورق وجفافه لضمان نجاح عملية إعادة تدويره.',
    'Plastic': '🛍️ تأكد من تنظيف العبوات البلاستيكية من السوائل قبل فرزها.'
}

def get_user_rank(points):
    if points < 50: return "🌱 مبتدئ أخضر", "أمامك خطوة لتصبح ناشطاً بيئياً!"
    elif points < 150: return "🌿 ناشط بيئي نشط", "أداء رائع! استمر في تدوير المخلفات."
    else: return "🏆 بطل الاستدامة الخارقة", "أنت نموذج مثالي لحماية البيئة!"

db = load_database()

if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = None

if st.session_state.logged_in_user is None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #28a745;'>♻️ EcoSort</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 25px;'>منصة الذكاء الاصطناعي لإدارة وإعادة تدوير المخلفات</p>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 تسجيل الدخول", "✨ إنشاء حساب جديد"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            login_user = st.text_input("اسم المستخدم", key="login_user")
            login_pass = st.text_input("كلمة المرور", type="password", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("تسجيل الدخول"):
                if login_user in db and db[login_user]["password"] == login_pass:
                    st.session_state.logged_in_user = login_user
                    st.rerun()
                else:
                    st.error("❌ اسم المستخدم أو كلمة المرور غير صحيحة!")
                    
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user = st.text_input("اختر اسم مستخدم جديد", key="new_user")
            new_pass = st.text_input("اختر كلمة المرور", type="password", key="new_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("إنشاء الحساب الان"):
                if new_user and new_pass:
                    if new_user in db:
                        st.error("⚠️ اسم المستخدم موجود مسبقاً، اختر اسماً آخر.")
                    else:
                        db[new_user] = {"password": new_pass, "points": 0, "items": 0, "co2": 0.0}
                        save_database(db)
                        st.success("🎉 تم إنشاء الحساب بنجاح! انتقل لتبويب تسجيل الدخول.")
                else:
                    st.warning("⚠️ برجاء ملء جميع الخانات المطلوبة.")
    st.stop()

current_user = st.session_state.logged_in_user
user_data = db[current_user]
rank_title, rank_msg = get_user_rank(user_data["points"])

st.sidebar.image("https://img.icons8.com/color/96/environmental-care.png", width=80)
st.sidebar.title(f"أهلاً بك، {current_user} 👋")

if st.sidebar.button("🚪 تسجيل الخروج"):
    st.session_state.logged_in_user = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.title("لوحة الأثر البيئي 🌍")
st.sidebar.markdown(f"""
    <div class="rank-card">
        <h4 style='color: #2e7d32; margin: 0;'>{rank_title}</h4>
        <p style='font-size: 12px; color: #388e3c; margin: 5px 0 0 0;'>{rank_msg}</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.metric(label="🏆 إجمالي النقاط", value=f"{user_data['points']} نقطة")
st.sidebar.metric(label="♻️ القطع المُعاد تدويرها", value=f"{user_data['items']} قطعة")
st.sidebar.metric(label="💨 غاز CO2 المُوفر", value=f"{user_data['co2']:.1f} جرام")

st.title(f"♻️ EcoSort: مرحباً بك يا {current_user}")
st.markdown("ارفع صورة للمخلفات من جهازك، دع الذكاء الاصطناعي يفحصها، واجمع نقاطك البيئية! 🌱")
st.markdown("---")

uploaded_file = st.file_uploader("📁 اختر صورة للمخلفات (JPG, PNG, JPEG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("### 📷 الصورة المُحملة")
        image_obj = Image.open(uploaded_file)
        st.image(image_obj, use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 نتائج الفحص البيئي")
        with st.spinner("🤖 جاري فحص الصورة وتحديد نوع النفايات..."):
            img = image_obj.convert('RGB').resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
        
        # عرض اسم النتيجة فقط بدون نسبة الثقة أو شريط التقدم
        st.markdown(f"""
            <div class="prediction-card">
                <h3 style='color: #28a745; margin: 0;'>النوع المكتشف: {predicted_class}</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.info(f"**نصيحة بيئية:** {recycling_tips.get(predicted_class, 'حافظ على نظافة بيئتك!')}")
        
        earned_points = points_mapping.get(predicted_class, 10)
        saved_co2 = co2_mapping.get(predicted_class, 30.0)
        
        if st.button(f"♻️ تأكيد إعادة التدوير وكسب (+{earned_points} نقطة)"):
            db[current_user]["points"] += earned_points
            db[current_user]["items"] += 1
            db[current_user]["co2"] += saved_co2
            save_database(db)
            
            st.success(f"ممتاز! أُضيف رصيد {earned_points} نقطة وحُفظت في حسابك بنجاح 🌿")
            st.balloons()
            st.rerun()

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 13px;'>تم التطوير لدعم الابتكار الأخضر ومشاريع التخرج المستدامة 🌿</p>", unsafe_allow_html=True)