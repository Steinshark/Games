@echo off 
DOSKEY ls=dir 
DOSKEY ml=C: $T cd C:\gitrepos\projects\ml $T conda activate music
DOSKEY gitwrk=git add . $T git commit -m $* $T git push
