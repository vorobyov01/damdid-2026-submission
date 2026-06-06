# Исправление проблемы с CUDA и multiprocessing

## Проблема

При запуске распределенного теста возникала ошибка:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. 
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

## Причина

По умолчанию Python multiprocessing на Linux использует метод `fork()` для создания новых процессов. Однако CUDA не может быть реинициализирована в форкнутом процессе, что приводит к ошибке.

## Решение

Исправлен файл `examples/simple/test_tp_verification.py`:

1. **Установка метода 'spawn'**: Добавлен код для установки метода запуска процессов на 'spawn' перед импортом torch:
```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

2. **Использование spawn контекста**: При создании процессов используется spawn контекст:
```python
ctx = mp.get_context('spawn')
p = ctx.Process(target=run_worker, args=(rank, world_size))
```

3. **Обработка ошибок**: Добавлена проверка успешного завершения всех процессов.

## Альтернативное решение

Для распределенных вычислений рекомендуется использовать `torchrun`, который правильно обрабатывает все аспекты распределенного запуска:

```bash
torchrun --nproc_per_node=2 test_tp_torchrun.py
```

Создан отдельный файл `test_tp_torchrun.py`, который оптимизирован для использования с `torchrun`.

## Изменения в коде

1. `test_tp_verification.py`: Исправлена работа с multiprocessing для CUDA
2. `test_tp_torchrun.py`: Новый файл для запуска через torchrun (рекомендуемый способ)

## Тестирование

После исправлений код должен работать корректно:
- В однопроцессном режиме: `python3 test_tp_verification.py`
- В многопроцессном режиме: `python3 test_tp_verification.py` (если доступно 2+ GPU)
- С torchrun: `torchrun --nproc_per_node=2 test_tp_torchrun.py`




