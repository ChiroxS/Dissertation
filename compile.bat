nvcc -rdc=true %~dp0\src\memcached.cu -o memcached ^
    -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\cl.exe"