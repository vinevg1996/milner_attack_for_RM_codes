# milner_attack_for_RM_codes

Перед запуском установите:
  
  pip3 install reedmuller
  pip3 install galois
  pip3 install numpy

Чтобы запусить в режиме 1 (генерация ключей), запустите:

  python3 main.py mode=1 r m secret_key.txt public_key.txt

Чтобы запусить в режиме 2 (поиск атаки), запустите:

  python3 main.py mode=2 public_key.txt r m recover_secret_key.txt

Чтобы запусить в режиме 3 (проверка), запустите:

  python3 main.py mode=3 recover_secret_key.txt public_key.txt r m

Пример запуска:
  
  python3 main.py mode=1 2 3 secret_key.txt public_key.txt
  python3 main.py mode=2 public_key.txt 2 3 recover_secret_key.txt
  python3 main.py mode=3 recover_secret_key.txt public_key.txt 2 3
