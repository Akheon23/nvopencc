CmdLine Visual Studio Add-In README
====================================

This add-in allows for a (more) convenient way to debug Open64 binaries
using the Visual Studio debugger.

To install, simply open the CmdLine.csproj project under Visual Studio,
and click 'Build Solution'.  The files
c/Program Files/Microsoft Visual Studio 8/Common7/IDE/devenvc/Program Files/Microsoft Visual Studio 8/Common7/IDE/devenv
   CmdLine.dll
   CmdLine.AddIn
   
shall be installed into the 

   %USERPROFILE%\My Documents\Visual Studio 2005\AddIns

folder.  After the add-in has been installed, you may debug any of the
Open64 executables (be.exe, gfec.exe, inline.exe, etc.) by issuing 
the command

  devenv /Command "CmdLine.Debug.Setup /path/name/of/open64.sln
                   /path/name/of/executable.exe arguments"

Possible Future Enhancements
----------------------------

The add-in could maintain a database of mappings from executables
being debugged to their corresponding Visual Studio projects/solutions.
For non-Makefile projects, it might also be possible to determine the
project/solution name directly from metadata found in the executable.

Also, the user should be able to supply a .vcproj instead of a .sln,
and the add-in should find the .sln using the same algorithm that
Visual Studio itself uses when searching for enclosing solutions.
