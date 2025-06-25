from collections import defaultdict


# Настройка уровня реакции на токсичность:
# 1 - только предупреждение
# 2 - мут на 60 секунд
# 3 - бан
chat_levels = defaultdict(lambda: 1)
