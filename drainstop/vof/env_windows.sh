# OpenFOAM v2606, native Windows build (OpenFOAM-v2606-windows-mingw.exe), for Git Bash.
# Set OF_ROOT to the install folder. Parallel runs need MS-MPI (thirdParty/msmpisetup.exe, admin)
# and the msmpi Pstream library:  cp libPstream.dll-msmpi libPstream.dll  in the bin folder below.
export OF_ROOT=${OF_ROOT:-/c/Users/laka/tools/OpenFOAM-v2606}
_of=$OF_ROOT/msys64/home/ofuser/OpenFOAM/OpenFOAM-v2606
export WM_PROJECT=OpenFOAM WM_PROJECT_VERSION=v2606 FOAM_SIGFPE=1
export WM_PROJECT_DIR="$(cygpath -w "$_of")"
export WM_THIRD_PARTY_DIR="$(cygpath -w "$_of/ThirdParty")"
export PATH="$_of/platforms/win64MingwDPInt32Opt/bin:$_of/ThirdParty/platforms/win64MingwDPInt32Opt/lib:$PATH"
[ -d "/c/Program Files/Microsoft MPI/Bin" ] && export PATH="/c/Program Files/Microsoft MPI/Bin:$PATH"
unset _of
