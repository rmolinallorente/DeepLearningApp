https://code.visualstudio.com/docs/cpp/config-mingw
pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain

https://packages.msys2.org/packages/mingw-w64-x86_64-gdb:
pacman -S mingw-w64-x86_64-gdb


Virtual E para que te cree una version 3.9 debes llamarlo desde Python39
1.-C:\Python39>py -m venv C:\RMolina\02Projects\U2_Flash\venv39

Py or python cuidado que coge versiones diferentes

2.-Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted -Force
3.-venvP39\Scripts>activate

4.-(venv39) PS C:\RMolina\02Projects\U2_Flash\utils> py -m pip install -r requirements.txt


Camaras:

Sin http--- usuario: root contraseña phadmin


esptool --chip esp32 --port COM7 erase_flash



nrfjprog --program merged.hex --sectoranduicrerase --verify -f NRF52