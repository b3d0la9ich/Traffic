from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime
from flask_migrate import Migrate
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from models import db, PassengerData, Prediction, User, Incident
# Следующие импорты не используются, можно удалить при желании
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:4780@db/traffic_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "supersecretkey"

db.init_app(app)
migrate = Migrate(app, db)

# 🔐 Функция-помощник
def is_admin():
    return session.get("is_admin", False)

with app.app_context():
    db.create_all()

    # Создание администратора при первом запуске
    admin = User.query.filter_by(username="admin").first()
    if not admin:
        admin = User(username="admin", is_admin=True)
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
        print("✅ Админ создан: admin/admin123")

    # Синтетические данные, если пусто
    if PassengerData.query.count() == 0:
        print("📊 Инициализация синтетических данных...")
        years = list(range(2016, 2026))
        months = list(range(1, 13))
        seasonal_factor = {
            1: 70, 2: 75, 3: 80, 4: 85, 5: 95, 6: 110,
            7: 130, 8: 125, 9: 100, 10: 90, 11: 80, 12: 95
        }
        for y in years:
            for m in months:
                base = 90
                passengers = base + seasonal_factor.get(m, 0) + np.random.normal(0, 3)
                db.session.add(PassengerData(year=y, month=m, passengers=round(passengers, 1)))
        db.session.commit()
        print("✅ Синтетические данные добавлены.")


def apply_incidents_to_forecast(df, incidents):
    """
    Пример функции (если понадобится): применяет влияние происшествий к будущим значениям прогноза.
    """
    for incident in incidents:
        affected_year = incident.year
        impact = incident.impact
        df.loc[
            (df['year'] == affected_year) & (df['date'] > pd.Timestamp.today()),
            'passengers'
        ] *= (1 + impact)
    return df


# ---------- Маршруты ----------

@app.route('/')
def index():
    return render_template('index.html')


# Регистрация
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        raw_username = request.form.get('username', '')
        raw_password = request.form.get('password', '')

        username = raw_username.strip()
        password = raw_password  # не .strip(), но проверим пробелы ниже

        # Пусто?
        if not username or not password:
            flash("Логин и пароль не могут быть пустыми.", "auth_danger")
            return render_template('register.html')

        # Запрет пробельных символов
        if any(ch.isspace() for ch in username) or any(ch.isspace() for ch in password):
            flash("Логин и пароль не должны содержать пробелы.", "auth_danger")
            return render_template('register.html')

        # Минимальные длины
        if len(username) < 3 or len(password) < 8:
            flash("Логин — минимум 3 символа, пароль — минимум 8 символов.", "auth_danger")
            return render_template('register.html')

        # Уникальность
        if User.query.filter_by(username=username).first():
            flash("Имя пользователя уже занято.", "auth_danger")
            return render_template('register.html')

        # Создание пользователя
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash("Регистрация успешна! Теперь войдите.", "auth_success")
        return redirect(url_for("login"))

    return render_template('register.html')


# Вход
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session.clear()
            session['user_id'] = user.id
            session['username'] = username
            session['is_admin'] = user.is_admin
            flash("Вы успешно вошли!", "auth_success")
            return redirect(url_for("dashboard"))
        else:
            flash("Неверное имя пользователя или пароль.", "auth_danger")

    return render_template('login.html')


# Выход
@app.route('/logout')
def logout():
    session.clear()
    flash("Вы вышли из аккаунта.", "auth_info")
    return redirect(url_for("login"))


# Панель
@app.route('/dashboard')
def dashboard():
    if "user_id" not in session:
        flash("Войдите в аккаунт.", "auth_warning")
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    return render_template('dashboard.html', user=user, username=user.username)


# Редактирование данных пассажиропотока
@app.route('/edit', methods=['GET', 'POST'])
def edit_data():
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            month = int(request.form['month'])
            passengers = int(request.form['passengers'])

            # Разрешаем редактировать только с 2016-01 по 2025-09
            if not (2016 <= year <= 2025):
                flash("Год должен быть от 2016 до 2025.", "edit_danger")
                return redirect(url_for('edit_data'))

            if not (1 <= month <= 12):
                flash("Месяц должен быть от 1 до 12.", "edit_danger")
                return redirect(url_for('edit_data'))

            if year == 2025 and month > 9:
                flash("Редактирование доступно только до сентября 2025 включительно.", "edit_danger")
                return redirect(url_for('edit_data'))

            if passengers < 0:
                flash("Количество пассажиров не может быть отрицательным.", "edit_danger")
                return redirect(url_for('edit_data'))

            entry = PassengerData.query.filter_by(year=year, month=month).first()
            if not entry:
                flash(f"Данные за {year}-{month:02d} не найдены.", "edit_warning")
                return redirect(url_for('edit_data'))

            entry.passengers = passengers
            db.session.commit()
            flash(f"Данные за {year}-{month:02d} обновлены.", "edit_success")
            return redirect(url_for('dashboard'))

        except ValueError:
            flash("Проверьте корректность введённых значений.", "edit_danger")
            return redirect(url_for('edit_data'))

    # --- GET: подготовим данные для предпросмотра и таблицы ---
    records = PassengerData.query.order_by(PassengerData.year, PassengerData.month).all()
    data_map = {f"{r.year}-{r.month:02d}": r.passengers for r in records}
    return render_template('edit_data.html', records=records, data_map=data_map)



# Прогноз
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    plot_url = None

    if request.method == 'POST':
        try:
            year = int(request.form['year'].strip())
            month = int(request.form['month'].strip())

            if not (2016 <= year <= 2030) or not (1 <= month <= 12):
                flash("Введите год 2016–2030 и месяц 1–12.", "pred_danger")
                return redirect(url_for("predict"))

            # Данные
            data = PassengerData.query.all()
            incidents = Incident.query.all()

            rows = [{'year': d.year, 'month': d.month, 'passengers': d.passengers} for d in data]
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

            # Признаки сезонности
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

            # Обучение
            X = df[['year', 'sin_month', 'cos_month']]
            y = df['passengers']
            model = LinearRegression().fit(X, y)

            # Точка прогноза
            pred_date = pd.to_datetime(f"{year}-{month:02d}-01", format="%Y-%m-%d")
            prediction = float(model.predict([[year,
                                               np.sin(2 * np.pi * month / 12),
                                               np.cos(2 * np.pi * month / 12)]])[0])

            # Сохраним прогноз
            db.session.add(Prediction(year=year, month=month, predicted_passengers=int(prediction)))
            db.session.commit()

            # График
            plt.figure(figsize=(10, 4))
            plt.plot(df['date'], df['passengers'], marker='o', label='Факт')
            plt.scatter([pred_date], [prediction], color='red', label='Прогноз', zorder=5)
            plt.legend()
            plt.title('Пассажиропоток с точкой прогноза')
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

        except Exception as e:
            flash(f"Ошибка при обработке прогноза: {e}", "pred_danger")
            return redirect(url_for("predict"))

    return render_template('predict.html', prediction=prediction, plot_url=plot_url)


# Инциденты
@app.route('/incidents', methods=['GET', 'POST'])
def incidents():
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            month = int(request.form['month'])
            duration = int(request.form['duration'])
            impact = float(request.form['impact'])
            description = (request.form.get('description') or '').strip()
        except ValueError:
            flash("Неверный формат полей.", "inc_danger")
            return redirect(url_for('incidents'))

        # Ограничение длины описания
        if len(description) > 255:
            flash(f"Описание слишком длинное ({len(description)}/255). Сократите текст.", "inc_danger")
            return redirect(url_for('incidents'))

        if not (2026 <= year <= 2030):
            flash("Год должен быть от 2026 до 2030.", "inc_danger")
            return redirect(url_for('incidents'))

        incident = Incident(
            year=year,
            month=month,
            duration=duration,
            impact=impact,
            description=description if description else None
        )
        db.session.add(incident)
        db.session.commit()
        flash("Инцидент добавлен.", "inc_success")
        return redirect(url_for('incidents'))

    all_incidents = Incident.query.order_by(Incident.year, Incident.month).all()
    return render_template("incidents.html", incidents=all_incidents)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
