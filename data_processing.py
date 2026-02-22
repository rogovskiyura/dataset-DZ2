import pandas as pd
import os

# Путь к файлу (измените при необходимости)
FILE_PATH = r"datasetSnowsales2.csv"


def check_missing_values():
    """
    Подсчитывает количество пропущенных значений в каждом столбце файла

    Returns:
    tuple: (DataFrame с данными, Series с количеством пропусков)
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"Файл {FILE_PATH} не найден")

        # Загружаем данные
        df = pd.read_csv(FILE_PATH, delimiter=';')

        # Подсчет пропущенных значений
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100

        # Создаем сводную таблицу
        missing_df = pd.DataFrame({
            'Кол-во пропусков': missing,
            'Процент': missing_percent
        })

        return df, missing_df

    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None, None


def generate_missing_report():
    """
    Генерирует подробный отчет о пропущенных значениях
    Выводит статистику для столбцов с пропусками
    """
    df, missing_df = check_missing_values()

    if df is None:
        return

    print("=" * 70)
    print("ОТЧЕТ О ПРОПУЩЕННЫХ ЗНАЧЕНИЯХ")
    print("=" * 70)

    # Фильтруем только столбцы с пропусками
    cols_with_missing = missing_df[missing_df['Кол-во пропусков'] > 0]

    if cols_with_missing.empty:
        print("Пропущенные значения не найдены.")
        return

    print(f"Всего записей: {len(df)}")
    print(f"Столбцы с пропусками: {len(cols_with_missing)}\n")

    # Для каждого столбца с пропусками выводим статистику
    for col in cols_with_missing.index:
        print(f"\n{'-' * 50}")
        print(f"СТОЛБЕЦ: '{col}'")
        print(f"{'-' * 50}")

        missing_count = cols_with_missing.loc[col, 'Кол-во пропусков']
        missing_percent = cols_with_missing.loc[col, 'Процент']
        print(f"Пропущено значений: {missing_count} ({missing_percent:.2f}%)")

        # Получаем данные без пропусков
        non_missing = df[col].dropna()

        if len(non_missing) > 0:
            # Определяем тип данных
            if pd.api.types.is_numeric_dtype(non_missing):
                # Для числовых данных
                print(f"\nСтатистика по имеющимся данным:")
                print(f"  Среднее: {non_missing.mean():.2f}")
                print(f"  Медиана: {non_missing.median():.2f}")
                print(f"  Минимум: {non_missing.min()}")
                print(f"  Максимум: {non_missing.max()}")

                # Наиболее частые значения (топ-3)
                value_counts = non_missing.value_counts().head(3)
                print(f"  Наиболее частые значения:")
                for val, count in value_counts.items():
                    print(f"    {val}: {count} раз ({count / len(non_missing) * 100:.1f}%)")

            else:
                # Для текстовых/категориальных данных
                print(f"\nСтатистика по имеющимся данным:")

                # Наиболее частые значения
                value_counts = non_missing.value_counts().head(5)
                print(f"  Наиболее частые значения:")
                for val, count in value_counts.items():
                    print(f"    '{val}': {count} раз ({count / len(non_missing) * 100:.1f}%)")

                # Уникальные значения
                print(f"  Уникальных значений: {non_missing.nunique()}")

        else:
            print("  Нет данных для анализа (все значения пропущены)")

    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ ПО ЗАПОЛНЕНИЮ")
    print("=" * 70)
    print("Для заполнения пропусков можно использовать:")
    print("- Числовые данные: среднее, медиана")
    print("- Категориальные данные: наиболее частое значение (мода)")
    print("- Временные ряды: интерполяция или предыдущее значение")


def main():
    """Основная функция для запуска анализа"""
    generate_missing_report()


if __name__ == "__main__":
    main()