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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:4780@db/traffic_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "supersecretkey"

db.init_app(app)
migrate = Migrate(app, db)

# 🔐 Админ-проверка
def is_admin():
    return session.get("is_admin", False)

with app.app_context():
    db.create_all()

    print("👤 Проверка администратора...")
    admin = User.query.filter_by(username="admin").first()
    if not admin:
        admin = User(username="admin", is_admin=True)
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
        print("✅ Админ создан: admin/admin123")
    else:
        print("⚠️ Админ уже существует.")

    # Добавим синтетические данные в привычном виде test_data
    if PassengerData.query.count() == 0:
        print("📊 Добавляем синтетические данные...")
        test_data = []
        years = list(range(2016, 2026))
        months = list(range(1, 13))
        for y in years:
            for m in months:
                base = 90
                seasonal_factor = {
                    1: 70, 2: 75, 3: 80, 4: 85, 5: 95, 6: 110,
                    7: 130, 8: 125, 9: 100, 10: 90, 11: 80, 12: 95
                }
                passengers = base + seasonal_factor.get(m, 0) + np.random.normal(0, 3)
                test_data.append((y, m, round(passengers, 1)))

        for year, month, passengers in test_data:
            db.session.add(PassengerData(year=year, month=month, passengers=passengers))
        db.session.commit()
        print("✅ Синтетические данные (в формате test_data) добавлены.")

    # Вывод всех существующих данных при запуске страницы
    all_data = PassengerData.query.order_by(PassengerData.year, PassengerData.month).all()
    print("\n📋 Все данные в базе:")
    for record in all_data:
        print(f"{record.year}-{record.month:02d}: {record.passengers} пассажиров")


def apply_incidents_to_forecast(df, incidents):
    """
    Применяет влияние происшествий к будущим значениям прогноза.
    Каждое происшествие влияет на весь год, к которому оно относится.
    """
    for incident in incidents:
        affected_year = incident.year
        impact = incident.impact

        df.loc[
            (df['year'] == affected_year) & (df['date'] > pd.Timestamp.today()),
            'passengers'
        ] *= (1 + impact)

    return df


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash("Ошибка: имя пользователя уже занято!", "danger")
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash("Регистрация успешна! Войдите в аккаунт.", "success")
            return redirect(url_for("login"))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session.clear()
            session['user_id'] = user.id
            session['username'] = username
            session['is_admin'] = user.is_admin
            flash("Вы успешно вошли!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Неверное имя пользователя или пароль.", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Вы вышли из аккаунта", "info")
    return redirect(url_for("login"))

@app.route('/dashboard')
def dashboard():
    if "user_id" not in session:
        flash("Войдите в аккаунт", "warning")
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    return render_template('dashboard.html', user=user, username=user.username)

@app.route('/edit', methods=['GET', 'POST'])
def edit_data():
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            month = int(request.form['month'])
            passengers = int(request.form['passengers'])

            # Проверки
            if not (2016 <= year <= 2030):
                flash("Год должен быть в диапазоне 2016–2030", "danger")
                return redirect(url_for('edit_data'))

            if not (1 <= month <= 12):
                flash("Месяц должен быть от 1 до 12", "danger")
                return redirect(url_for('edit_data'))

            if passengers < 0:
                flash("Количество пассажиров не может быть отрицательным", "danger")
                return redirect(url_for('edit_data'))

            # Поиск записи
            entry = PassengerData.query.filter_by(year=year, month=month).first()
            if not entry:
                flash(f"Данные за {year}-{month:02d} не найдены!", "warning")
                return redirect(url_for('edit_data'))

            # Обновление
            entry.passengers = passengers
            db.session.commit()
            flash(f"Данные за {year}-{month:02d} успешно обновлены!", "success")
            return redirect(url_for('dashboard'))

        except ValueError:
            flash("Проверьте правильность введённых значений", "danger")
            return redirect(url_for('edit_data'))

    return render_template('edit_data.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    plot_url = None

    if request.method == 'POST':
        try:
            year = int(request.form['year'].strip())
            month = int(request.form['month'].strip())

            if not (2016 <= year <= 2030) or not (1 <= month <= 12):
                flash("Введите год от 2016 до 2030 и месяц от 1 до 12!", "danger")
                return redirect(url_for("predict"))

            # Получение данных
            data = PassengerData.query.all()
            incidents = Incident.query.all()

            rows = [{'year': d.year, 'month': d.month, 'passengers': d.passengers} for d in data]
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

            # Обработка инцидентов с учётом продолжительности и постепенного восстановления
            incident_effects = pd.Series(1.0, index=pd.date_range(start='2026-01-01', end='2030-12-01', freq='MS'))

            for inc in incidents:
                start = pd.to_datetime(f"{inc.year}-{inc.month:02d}-01", format="%Y-%m-%d")
                for i in range(inc.duration):
                    month_inc = start + pd.DateOffset(months=i)
                    if month_inc in incident_effects.index:
                        decay = 1 + inc.impact * (1 - i / inc.duration)
                        incident_effects[month_inc] *= decay

            # Добавляем признаки
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

            # Предсказываемая дата
            pred_date = pd.to_datetime(f"{year}-{month:02d}-01", format="%Y-%m-%d")

            X = df[['year', 'sin_month', 'cos_month']]
            y = df['passengers']

            model = LinearRegression()
            model.fit(X, y)

            start_date = pred_date - pd.DateOffset(years=2)
            full_range = pd.date_range(start=start_date, end=pred_date, freq='MS')

            df_full = pd.DataFrame({'date': full_range})
            df_full['year'] = df_full['date'].dt.year
            df_full['month'] = df_full['date'].dt.month
            df_full['sin_month'] = np.sin(2 * np.pi * df_full['month'] / 12)
            df_full['cos_month'] = np.cos(2 * np.pi * df_full['month'] / 12)

            df_full = pd.merge(df_full, df[['date', 'passengers']], on='date', how='left')

            missing = df_full['passengers'].isna()
            if missing.any():
                X_missing = df_full.loc[missing, ['year', 'sin_month', 'cos_month']]
                predicted = model.predict(X_missing)
                df_full.loc[missing, 'passengers'] = predicted

            # Применяем влияние инцидентов
            df_full['adjusted'] = df_full['passengers']
            for date in df_full['date']:
                if date in incident_effects.index:
                    df_full.loc[df_full['date'] == date, 'adjusted'] *= incident_effects[date]

            prediction = df_full.loc[df_full['date'] == pred_date, 'adjusted'].values[0]
            pred = Prediction(year=year, month=month, predicted_passengers=int(prediction))
            db.session.add(pred)
            db.session.commit()

            # Построение графика
            plt.figure(figsize=(10, 4))
            is_real = df_full['date'].isin(df['date'])
            is_pred = ~is_real

            plt.plot(df_full.loc[is_real, 'date'], df_full.loc[is_real, 'adjusted'],
                     marker='o', label='Факт')
            plt.plot(df_full.loc[is_pred, 'date'], df_full.loc[is_pred, 'adjusted'],
                     marker='o', linestyle='dashed', color='blue', label='Прогноз')

            plt.scatter([pred_date], [prediction], color='red', label='Прогноз (текущий)', zorder=5)
            plt.annotate(f'{prediction:.0f}', xy=(pred_date, prediction), xytext=(5, 5),
                         textcoords='offset points', color='red')

            plt.xticks(df_full['date'], df_full['date'].dt.strftime('%Y-%m'), rotation=45)
            plt.title('Пассажиропоток с прогнозом')
            plt.ylabel('Количество пассажиров')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

        except Exception as e:
            flash(f"Ошибка при обработке прогноза: {str(e)}", "danger")
            return redirect(url_for("predict"))

    return render_template('predict.html', prediction=prediction, plot_url=plot_url)


@app.route('/incidents', methods=['GET', 'POST'])
def incidents():
    if request.method == 'POST':
        year = int(request.form['year'])
        month = int(request.form['month'])
        duration = int(request.form['duration'])
        impact = float(request.form['impact'])
        description = request.form['description']

        if year < 2026 or year > 2030:
            flash("Год должен быть от 2026 до 2030", "danger")
        else:
            incident = Incident(year=year, month=month, duration=duration, impact=impact, description=description)
            db.session.add(incident)
            db.session.commit()
            flash("Инцидент добавлен", "success")
        return redirect(url_for('incidents'))

    all_incidents = Incident.query.order_by(Incident.year, Incident.month).all()
    return render_template("incidents.html", incidents=all_incidents)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
