import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Класс для загрузки данных о продажах снега из CSV файла.
    """

    # Явное указание пути к файлу - измените при необходимости
    # Можно использовать абсолютный путь: r"C:\Users\Username\Downloads\DatasetSnowSales2.csv"
    # Или относительный путь от текущей директории
    DEFAULT_FILE_PATH = Path(__file__).parent / "DatasetSnowSales2.csv"

    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        """
        Инициализация загрузчика данных.

        Args:
            file_path: Путь к CSV файлу. Если не указан, используется путь по умолчанию.
        """
        self.file_path = Path(file_path) if file_path else self.DEFAULT_FILE_PATH
        self.data: Optional[pd.DataFrame] = None

    def load_data(self, delimiter: str = ';') -> Optional[pd.DataFrame]:
        """
        Загрузка данных из CSV файла.

        Args:
            delimiter: Разделитель полей в CSV (по умолчанию ';')

        Returns:
            DataFrame с загруженными данными или None в случае ошибки
        """
        try:
            # Проверка существования файла
            if not self.file_path.exists():
                logger.error(f"Файл не найден: {self.file_path}")
                logger.info(f"Текущая рабочая директория: {Path.cwd()}")
                return None

            logger.info(f"Загрузка данных из файла: {self.file_path}")

            # Загрузка CSV с правильной обработкой пропущенных значений
            self.data = pd.read_csv(
                self.file_path,
                delimiter=delimiter,
                encoding='utf-8',
                na_values=['', 'NA', 'NULL'],  # Указание значений, которые считать NaN
                keep_default_na=True
            )

            # Базовая очистка данных
            self._clean_data()

            # Вывод информации о загруженных данных
            logger.info(f"Данные успешно загружены. Размер: {self.data.shape}")
            logger.info(f"Колонки: {list(self.data.columns)}")

            return self.data

        except pd.errors.EmptyDataError:
            logger.error("Файл пуст")
            return None
        except pd.errors.ParserError as e:
            logger.error(f"Ошибка парсинга CSV: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке данных: {e}")
            return None

    def _clean_data(self) -> None:
        """
        Внутренний метод для первичной очистки данных.
        """
        if self.data is None:
            return

        # Удаление пробелов из названий колонок
        self.data.columns = self.data.columns.str.strip()

        # Обработка пропущенных значений в числовых колонках
        numeric_columns = ['sales', 'tempday', 'tempnight']
        for col in numeric_columns:
            if col in self.data.columns:
                # Преобразование в числовой тип, ошибки преобразования станут NaN
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Преобразование даты
        if 'Date' in self.data.columns:
            try:
                self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d.%m.%Y', errors='coerce')
            except Exception as e:
                logger.warning(f"Не удалось преобразовать даты: {e}")

        # Стандартизация типов осадков (приведение к нижнему регистру)
        if 'precipitation' in self.data.columns:
            self.data['precipitation'] = self.data['precipitation'].str.lower()

    def get_summary_stats(self) -> dict:
        """
        Получение базовой статистики по данным.

        Returns:
            Словарь со статистикой или пустой словарь, если данные не загружены
        """
        if self.data is None:
            logger.warning("Данные не загружены. Сначала вызовите load_data()")
            return {}

        stats = {
            'total_rows': len(self.data),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.astype(str).to_dict()
        }

        # Статистика по числовым колонкам
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats['numeric_stats'] = self.data[numeric_cols].describe().to_dict()

        return stats

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Получение загруженных данных.

        Returns:
            DataFrame с данными или None, если данные не загружены
        """
        if self.data is None:
            logger.warning("Данные не загружены. Сначала вызовите load_data()")
        return self.data


# Пример использования модуля
if __name__ == "__main__":
    # Создание экземпляра загрузчика
    loader = DataLoader()

    # Загрузка данных
    df = loader.load_data()

    if df is not None:
        print("\nПервые 5 строк данных:")
        print(df.head())

        print("\nИнформация о данных:")
        print(df.info())

        print("\nБазовая статистика:")
        stats = loader.get_summary_stats()
        print(f"Всего строк: {stats.get('total_rows')}")
        print(f"Колонки: {stats.get('columns')}")
    else:
        print("Не удалось загрузить данные. Проверьте путь к файлу.")