Building Open64 with Visual Studio 2005
=======================================

As of this writing, under Win32 one may only build Open64 using the Cygwin
environment.  However, it is possible to instruct the build system to use
the Microsoft Visual Studio 2005 compilers and linkers instead of gcc/ld.
Simply create the following environment variable:

   export USE_MSVC8=1

Then, build Open64 as you normally would.  (NOTE: For some reason, Cygwin
command-line builds sometimes fail if a copy of Visual Studio 2005 is
running on your machine, so you may need to exit the IDE first.)

Note that Open64 binaries built with Visual Studio shall be placed
underneath the

   targia32vs_nvisa

subdirectory.  (The MinGW gcc-based builds continue to use the
targia32gw_nvisa subdirectory.)  As of this writing, Visual Studio use
is not supported for Release builds, or for Win64 builds.

The resulting binaries can be debugged directly in the Visual Studio 2005
IDE.  From the Cygwin shell, this can be accomplished by the command:

   /path/to/devenv.exe /debugexe /path/to/open64.executable.exe <arguments>

In order to take advantage of IntelliSense, one must use the 'open64.sln'
solution (found in this directory) before starting a debug session.
To simplify things a bit further, I wrote a simple Visual Studio add-in
(see the AddIn/README.TXT file for more information) which allows for IDE
debug sessions using 'open64.sln' (or any other user-specified solution)
to be launched from the command line.




Zem Laski
18 September 2008

