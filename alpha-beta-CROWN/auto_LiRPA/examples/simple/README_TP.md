# Tensor Parallelism для auto_LiRPA

Это первая версия реализации Tensor Parallelism (TP) для библиотеки auto_LiRPA, основанная на исследовании из документа `gemini.out`.

## Что реализовано

1. **Классы для TP**: `BoundLinearTP_Col` и `BoundLinearTP_Row` в `auto_LiRPA/operators/tensor_parallel.py`
   - `BoundLinearTP_Col`: Column Parallel слои (расширяют размерность)
   - `BoundLinearTP_Row`: Row Parallel слои (сжимают размерность)

2. **Тестовый скрипт**: `test_tp_verification.py` - демонстрирует базовую верификацию

3. **Модель с TP**: `tp_model.py` - пример модели с TP слоями

## Как использовать

### Базовый пример (без TP)

```bash
cd /workspace/auto_LiRPA/examples/simple
python3 test_tp_verification.py
```

Это запустит простой пример верификации на одной GPU/CPU.

### Распределенный пример (с TP)

Для полного тестирования TP требуется минимум 2 GPU:

```bash
torchrun --nproc_per_node=2 test_tp_verification.py
```

## Архитектура

### Column Parallel (BoundLinearTP_Col)
- Веса разделены по выходной размерности
- В обратном проходе CROWN: входящие матрицы A разделены
- Требуется AllReduce для объединения результатов

### Row Parallel (BoundLinearTP_Row)
- Веса разделены по входной размерности  
- В обратном проходе CROWN: входящие матрицы A реплицированы
- Результат автоматически разделен (без коммуникации)

## Математические основы

Основная операция CROWN: `A_prev = A_curr @ W`

Для Column Parallel:
- `A_curr` разделена: `[A_curr_0, A_curr_1]`
- `W` разделена: `[W_0; W_1]`
- Локально: `partial_A = A_curr_i @ W_i`
- AllReduce: `A_prev = sum(partial_A)`

Для Row Parallel:
- `A_curr` реплицирована (полная матрица)
- `W` разделена: `[W_0, W_1]`
- Локально: `A_prev_i = A_curr @ W_i`
- Результат автоматически разделен

## Ограничения текущей версии

1. **Упрощенная реализация**: Текущая версия демонстрирует концепцию, но требует доработки для полной интеграции с auto_LiRPA
2. **Шардирование весов**: Модель должна быть правильно инициализирована с разделенными весами
3. **Обработка активаций**: Необходимо правильно обрабатывать разделенные активации между слоями

## Следующие шаги

1. Полная интеграция с графом вычислений auto_LiRPA
2. Автоматическое определение типа параллелизма (Col/Row) на основе структуры модели
3. Оптимизация коммуникаций (overlapping computation and communication)
4. Поддержка Pipeline Parallelism для очень глубоких сетей

## Ссылки

- Документ исследования: `gemini.out`
- Базовый пример: `toy.py`
- Документация auto_LiRPA: https://auto-lirpa.readthedocs.io


