
To create solution:

[1]  Create solution xxxLib in VS and the dir is …../HCIP/

HCIP/xxxLib/

 xxxEngine.cpp,    xxxEngine.h,   xxxLib.xcxproj,   xxxLib.xcxproj.filters,   xxxLib.xcxproj.user

[2]  HCIP/

     xxxLib.sln,    xxxLib.sln.props,     xxxLib_GetRefs.cmd

[3]  HCIP/LibSupport

     xxxLib.props

*  If you create Lib solution, it will create xxxEngine.sln,  xxxEngine.cpp,    xxxEngine.h,   xxxLib.xcxproj,   xxxLib.xcxproj.filters,   xxxLib.xcxproj.user,  framework.h,  pch.cpp, pch.h

To create solution with 3rd party library:

[1] M:\US07-Engineering\SoftDev\References\3rd Party\Libraries\CryptoPP\8.6.0

[1]

HCIP/xxxLib/

     xxxEngine.cpp,    xxxEngine.h,     xxxLib.xcxproj,    xxxLib.xcxproj.filters

HCIP/

     xxxLib.sln,    xxxLib.sln.props,     xxxLib_GetRefs.cmd

HCIP/LibSupport

     xxxLib.props