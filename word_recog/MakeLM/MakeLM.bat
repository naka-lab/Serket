:: 連続で実行するとパスが増えてエラーになるのでsetlocalで有効な範囲を限定する
setlocal

:: Pathの設定
set PATH=%~p0;%PATH%
set PATH=%~p0;C:\Users\kuniyasu\Anaconda2;

:: 一時ファイルの置き場所を作成
rd /s /q tmp
mkdir tmp

:: テキストを整形
python %~p0\preprocessText.py %1 > tmp/data.txt

::単語発生頻度を計算
text2wfreq tmp/data.txt tmp/mklm.wfreq

::単語リストを作成
wfreq2vocab -gt 0 tmp/mklm.wfreq tmp/mklm.vocab

::ID 3-gramを作成
text2idngram -vocab tmp/mklm.vocab < tmp/data.txt > tmp/mklm.id3gram

::ID 2-gramを作成
text2idngram -n 2 -vocab tmp/mklm.vocab < tmp/data.txt > tmp/mklm.id2gram

::3-gramを逆順にする
reverseidngram tmp/mklm.id3gram tmp/mklm.revid3gram

::3-gramモデルの作成
idngram2lm -idngram tmp/mklm.revid3gram -vocab tmp/mklm.vocab -arpa tmp/mklm.rev3gram.arpa

::2-gramモデルの作成
idngram2lm -n 2 -idngram tmp/mklm.id2gram -vocab tmp/mklm.vocab -arpa tmp/mklm.2gram.arpa

::1つの辞書にまとめる
mkbingram tmp/mklm.2gram.arpa tmp/mklm.rev3gram.arpa %2.bingram

::julius用の単語辞書を作成
python %~p0\vocab2htkdict.py  tmp/mklm.vocab > %2.htkdic

endlocal