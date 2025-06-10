#!/bin/bash

echo "👉 Jalankan training model..."
python -u model/train_model.py

echo "👉 Cek apakah model dan vectorizer sudah update:"
ls -lh model/*.pkl

echo "👉 Tambahkan model ke Git..."
git add model/*.pkl

echo "👉 Commit perubahan model..."
git commit -m "update trained model (auto via train_and_push.sh)"

echo "👉 Push ke GitHub..."
git push

echo "✅ Semua selesai. Siap cek di Azure nanti!"
