import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Настройка стилей для профессиональных графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11


class SnowSalesVisualizer:
    """
    Класс для визуализации данных о продажах в зимний период
    """

    def __init__(self, file_path: str = 'DatasetSnowSales2.csv'):
        """
        Инициализация с загрузкой данных
        """
        self.file_path = file_path
        self.df = None
        self.weather_colors = {
            'cloud': '#95a5a6',  # серый
            'rain': '#3498db',  # синий
            'snow': '#ecf0f1',  # белый/светло-серый
            'Sun': '#f1c40f'  # желтый
        }
        self.load_data()

    def load_data(self):
        """Загрузка и предобработка данных"""
        if not Path(self.file_path).exists():
            print(f"Файл не найден: {self.file_path}")
            return False

        try:
            # Загрузка с правильным разделителем (;)
            self.df = pd.read_csv(self.file_path, delimiter=';', encoding='utf-8')

            # Преобразование даты
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d.%m.%Y', errors='coerce')

            # Проверка на ошибки в датах
            if self.df['Date'].isnull().any():
                print(f"Обнаружены ошибки в датах. Проверьте формат.")
                self.df = self.df.dropna(subset=['Date'])

            # Обработка пропущенных значений в sales
            missing_sales = self.df['sales'].isnull().sum()
            print(f"Пропущенные значения в sales: {missing_sales}")

            if missing_sales > 0:
                # Заполним пропуск медианным значением для этого дня недели
                missing_indices = self.df[self.df['sales'].isnull()].index
                for idx in missing_indices:
                    missing_dayweek = self.df.loc[idx, 'dayweek']
                    median_sales = self.df[self.df['dayweek'] == missing_dayweek]['sales'].median()
                    self.df.loc[idx, 'sales'] = median_sales
                    print(f"Пропуск заполнен медианным значением для дня {missing_dayweek}: {median_sales:.0f}")

            # Преобразование типов
            self.df['sales'] = pd.to_numeric(self.df['sales'], errors='coerce')
            self.df['tempday'] = pd.to_numeric(self.df['tempday'], errors='coerce')
            self.df['tempnight'] = pd.to_numeric(self.df['tempnight'], errors='coerce')
            self.df['dayweek'] = pd.to_numeric(self.df['dayweek'], errors='coerce')

            # Добавим полезные колонки
            self.df['month'] = self.df['Date'].dt.month
            self.df['day'] = self.df['Date'].dt.day
            self.df['week'] = self.df['Date'].dt.isocalendar().week
            self.df['temp_avg'] = (self.df['tempday'] + self.df['tempnight']) / 2

            print(f"Данные загружены: {self.df.shape[0]} строк, {self.df.shape[1]} колонок")
            print(f"Период: с {self.df['Date'].min().date()} по {self.df['Date'].max().date()}")
            print(f"Диапазон продаж: {self.df['sales'].min():.0f} - {self.df['sales'].max():.0f}")

            return True

        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return False

    # =========================================================================
    # 1. ГИСТОГРАММЫ
    # =========================================================================

    def plot_histograms(self, columns: List[str] = None, save_path: Optional[str] = None):
        """
        Построение гистограмм для анализа распределений
        """
        if self.df is None:
            print("Данные не загружены")
            return

        if columns is None:
            columns = ['sales', 'tempday', 'tempnight', 'temp_avg']

        # Фильтруем только существующие колонки
        columns = [col for col in columns if col in self.df.columns]

        if not columns:
            print("Нет доступных колонок для визуализации")
            return

        # Создаем сетку графиков
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        for idx, (col, color) in enumerate(zip(columns, colors)):
            ax = axes[idx]
            data = self.df[col].dropna()

            if len(data) == 0:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
                continue

            # Гистограмма
            n, bins, patches = ax.hist(data, bins=20, density=False,
                                       alpha=0.7, color=color, edgecolor='black',
                                       linewidth=1.2)

            # Статистика
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()

            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                       label=f'Среднее: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='-.', linewidth=2,
                       label=f'Медиана: {median_val:.1f}')

            # Метрики
            stats_text = f'sigma={std_val:.1f}\nMin={data.min():.1f}\nMax={data.max():.1f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f'Распределение: {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Частота')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

        # Скрываем лишние графики
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Анализ распределений', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()

    # =========================================================================
    # 2. ЛИНЕЙНЫЕ ГРАФИКИ
    # =========================================================================

    def plot_time_series(self, save_path: Optional[str] = None):
        """
        Линейные графики временных рядов
        """
        if self.df is None:
            print("Данные не загружены")
            return

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # 1. Продажи по времени
        ax1 = axes[0]
        ax1.plot(self.df['Date'], self.df['sales'], 'o-', color='#3498db',
                 linewidth=2, markersize=4, label='Продажи')

        # Добавляем скользящее среднее
        if len(self.df) >= 7:
            rolling_mean = self.df['sales'].rolling(window=7, center=True).mean()
            ax1.plot(self.df['Date'], rolling_mean, 'r-', linewidth=3,
                     label='Скользящее среднее (7 дней)')

        # Отмечаем праздники
        holidays = ['2025-12-31', '2026-01-01', '2026-01-07']
        for holiday in holidays:
            ax1.axvline(pd.to_datetime(holiday), color='red', linestyle='--', alpha=0.5)

        ax1.set_title('Динамика продаж во времени', fontweight='bold')
        ax1.set_xlabel('Дата')
        ax1.set_ylabel('Продажи')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Температура днем и ночью
        ax2 = axes[1]
        ax2.plot(self.df['Date'], self.df['tempday'], 'o-', color='#e74c3c',
                 linewidth=2, markersize=4, label='Дневная температура')
        ax2.plot(self.df['Date'], self.df['tempnight'], 'o-', color='#3498db',
                 linewidth=2, markersize=4, label='Ночная температура')
        ax2.fill_between(self.df['Date'], self.df['tempnight'], self.df['tempday'],
                         alpha=0.2, color='gray')
        ax2.set_title('Температура днем и ночью', fontweight='bold')
        ax2.set_xlabel('Дата')
        ax2.set_ylabel('Температура (C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Осадки по дням
        ax3 = axes[2]
        weather_counts = self.df.groupby('Date')['precipitation'].first()
        colors = [self.weather_colors.get(w, '#95a5a6') for w in weather_counts.values]

        for date, weather in weather_counts.items():
            color = self.weather_colors.get(weather, '#95a5a6')
            ax3.axvline(date, color=color, alpha=0.5, linewidth=2)

        ax3.set_title('Тип осадков по дням', fontweight='bold')
        ax3.set_xlabel('Дата')
        ax3.set_ylabel('Тип осадков')
        ax3.set_yticks([])

        # Легенда для погоды
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=weather, alpha=0.7)
                           for weather, color in self.weather_colors.items()]
        ax3.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()

    # =========================================================================
    # 3. ДИАГРАММЫ РАССЕЯНИЯ
    # =========================================================================

    def plot_scatter(self, x_col: str = 'tempday', y_col: str = 'sales',
                     color_by: str = 'precipitation', save_path: Optional[str] = None):
        """
        Диаграмма рассеяния с цветовым кодированием по погоде
        """
        if self.df is None:
            print("Данные не загружены")
            return

        if x_col not in self.df.columns or y_col not in self.df.columns:
            print(f"Колонки {x_col} или {y_col} не найдены")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        # Создаем scatter plot с цветами по погоде
        unique_weather = self.df['precipitation'].dropna().unique()
        for weather in unique_weather:
            mask = self.df['precipitation'] == weather
            color = self.weather_colors.get(weather, '#95a5a6')

            ax.scatter(self.df.loc[mask, x_col], self.df.loc[mask, y_col],
                       c=color, label=weather, s=100, alpha=0.7,
                       edgecolors='black', linewidth=1)

        # Добавляем линию тренда
        clean_data = self.df[[x_col, y_col]].dropna()
        if len(clean_data) > 1:
            z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2.5, alpha=0.8,
                    label=f'Тренд: y = {z[0]:.2f}x + {z[1]:.2f}')

            # Корреляция
            corr = clean_data[x_col].corr(clean_data[y_col])
            ax.text(0.05, 0.95, f'Корреляция: {corr:.3f}', transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontweight='bold')

        ax.set_xlabel(x_col, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=14, fontweight='bold')
        ax.set_title(f'Зависимость продаж от {x_col}', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()

    # =========================================================================
    # 4. ДИАГРАММА ПО ДНЯМ НЕДЕЛИ
    # =========================================================================

    def plot_weekday_analysis(self, save_path: Optional[str] = None):
        """
        Анализ продаж по дням недели
        """
        if self.df is None:
            print("Данные не загружены")
            return

        # Словарь для дней недели
        weekdays = {1: 'Пн', 2: 'Вт', 3: 'Ср', 4: 'Чт', 5: 'Пт', 6: 'Сб', 7: 'Вс'}
        self.df['weekday_name'] = self.df['dayweek'].map(weekdays)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Средние продажи по дням недели
        ax1 = axes[0]
        weekday_sales = self.df.groupby('weekday_name')['sales'].agg(['mean', 'std']).reindex(
            ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])

        ax1.bar(weekday_sales.index, weekday_sales['mean'], yerr=weekday_sales['std'],
                capsize=10, color='#3498db', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('День недели')
        ax1.set_ylabel('Средние продажи')
        ax1.set_title('Средние продажи по дням недели', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Добавляем значения
        for i, (idx, row) in enumerate(weekday_sales.iterrows()):
            if not pd.isna(row['mean']):
                ax1.text(i, row['mean'] + 10, f'{row["mean"]:.0f}',
                         ha='center', fontweight='bold')

        # 2. Box plot продаж по дням недели
        ax2 = axes[1]
        df_weekday = self.df.dropna(subset=['weekday_name', 'sales']).copy()
        if not df_weekday.empty:
            df_weekday['weekday_name'] = pd.Categorical(df_weekday['weekday_name'],
                                                        categories=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'],
                                                        ordered=True)

            sns.boxplot(data=df_weekday, x='weekday_name', y='sales', ax=ax2, palette='Set3')
            ax2.set_xlabel('День недели')
            ax2.set_ylabel('Продажи')
            ax2.set_title('Распределение продаж по дням недели', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

        # 3. Количество наблюдений по дням недели
        ax3 = axes[2]
        day_counts = self.df['weekday_name'].value_counts().reindex(['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])

        ax3.bar(day_counts.index, day_counts.values, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('День недели')
        ax3.set_ylabel('Количество дней')
        ax3.set_title('Количество дней по типам', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        for i, (idx, val) in enumerate(day_counts.items()):
            if not pd.isna(val):
                ax3.text(i, val + 0.5, str(int(val)), ha='center', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()

    # =========================================================================
    # 5. ПОГОДНЫЙ АНАЛИЗ
    # =========================================================================

    def plot_weather_analysis(self, save_path: Optional[str] = None):
        """
        Анализ влияния погоды на продажи
        """
        if self.df is None:
            print("Данные не загружены")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Средние продажи по типу погоды
        ax1 = axes[0, 0]
        weather_sales = self.df.groupby('precipitation')['sales'].agg(['mean', 'std']).sort_values('mean',
                                                                                                   ascending=False)

        colors = [self.weather_colors.get(w, '#95a5a6') for w in weather_sales.index]
        ax1.bar(weather_sales.index, weather_sales['mean'], yerr=weather_sales['std'],
                capsize=10, color=colors, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Тип погоды')
        ax1.set_ylabel('Средние продажи')
        ax1.set_title('Средние продажи по типу погоды', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Распределение продаж по погоде (box plot)
        ax2 = axes[0, 1]
        df_weather = self.df.dropna(subset=['precipitation', 'sales'])
        if not df_weather.empty:
            sns.boxplot(data=df_weather, x='precipitation', y='sales', ax=ax2, palette=colors)
            ax2.set_xlabel('Тип погоды')
            ax2.set_ylabel('Продажи')
            ax2.set_title('Распределение продаж по типу погоды', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

        # 3. Зависимость продаж от температуры с учетом погоды
        ax3 = axes[1, 0]
        for weather in self.df['precipitation'].dropna().unique():
            mask = self.df['precipitation'] == weather
            color = self.weather_colors.get(weather, '#95a5a6')
            ax3.scatter(self.df.loc[mask, 'temp_avg'], self.df.loc[mask, 'sales'],
                        c=color, label=weather, s=80, alpha=0.7, edgecolors='black')

        ax3.set_xlabel('Средняя температура (C)')
        ax3.set_ylabel('Продажи')
        ax3.set_title('Зависимость продаж от температуры', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Количество дней по типам погоды
        ax4 = axes[1, 1]
        weather_counts = self.df['precipitation'].value_counts()
        if not weather_counts.empty:
            wedges, texts, autotexts = ax4.pie(weather_counts.values, labels=weather_counts.index,
                                               colors=[self.weather_colors.get(w, '#95a5a6') for w in weather_counts.index],
                                               autopct='%1.1f%%', startangle=90, textprops={'fontweight': 'bold'})
            ax4.set_title('Соотношение типов погоды', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()

    # =========================================================================
    # ИНТЕРАКТИВНОЕ МЕНЮ
    # =========================================================================

    def interactive_menu(self):
        """
        Интерактивное меню для выбора визуализаций
        """
        if self.df is None:
            print("Данные не загружены")
            return

        while True:
            print("\n" + "=" * 60)
            print("АНАЛИЗ ДАННЫХ О ПРОДАЖАХ В ЗИМНИЙ ПЕРИОД")
            print("=" * 60)
            print("\nДоступные визуализации:")
            print("1. Гистограммы (распределения sales, temperature)")
            print("2. Линейные графики (временные ряды)")
            print("3. Диаграммы рассеяния (зависимость от температуры)")
            print("4. Анализ по дням недели")
            print("5. Погодный анализ")
            print("0. Выход")

            choice = input("\nВаш выбор (0-5): ").strip()

            if choice == '1':
                print("\nГистограммы:")
                self.plot_histograms()

            elif choice == '2':
                print("\nЛинейные графики:")
                self.plot_time_series()

            elif choice == '3':
                print("\nДиаграммы рассеяния:")
                print("1. Продажи vs Дневная температура")
                print("2. Продажи vs Ночная температура")
                print("3. Продажи vs Средняя температура")
                scatter_choice = input("Выберите тип (1-3): ").strip()

                if scatter_choice == '1':
                    self.plot_scatter(x_col='tempday', y_col='sales')
                elif scatter_choice == '2':
                    self.plot_scatter(x_col='tempnight', y_col='sales')
                elif scatter_choice == '3':
                    self.plot_scatter(x_col='temp_avg', y_col='sales')
                else:
                    print("Неверный выбор")

            elif choice == '4':
                print("\nАнализ по дням недели:")
                self.plot_weekday_analysis()

            elif choice == '5':
                print("\nПогодный анализ:")
                self.plot_weather_analysis()

            elif choice == '0':
                print("\nДо свидания!")
                break

            else:
                print("\nНеверный выбор. Пожалуйста, выберите 0-5.")


# =============================================================================
# ЗАПУСК
# =============================================================================

if __name__ == "__main__":
    # Создаем визуализатор и запускаем интерактивное меню
    viz = SnowSalesVisualizer('DatasetSnowSales2.csv')

    if viz.df is not None:
        # Показываем базовую статистику
        print("\nБазовая статистика:")
        print(viz.df[['sales', 'tempday', 'tempnight']].describe().round(1))

        # Запускаем интерактивное меню
        viz.interactive_menu()
    else:
        print("Не удалось загрузить данные. Проверьте файл.")