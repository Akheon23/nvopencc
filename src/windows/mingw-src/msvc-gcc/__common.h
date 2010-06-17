#include <direct.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <windows.h>

#include <_mingw.h>

#define BIG_BUFFER 60000

static char *SCRIPT, *INCLUDE, *LIB, *PATH, *SYSDIR, *SYSTEMROOT;
static char *__THIS_DIR, *__MSVC_ROOT, *__OUT, *__CMD;
static char *__CL, *__LINK, *__BSCMAKE, *__STDOUT_FILE, *__STDOUT_MAP;
static char *__CL_ARGS, *__LINK_ARGS, *__CL_VERSION, *__LINK_VERSION;
static char *__BSCMAKE_VERSION, *__CL_DEBUG, *__LINK_DEBUG, *__BROWSE_FILES;
static char *__LIB_ARGS, *__LIB_VERSION, *__LIB, *__BSCMAKE_ARGS, *__DEPENDENCY_MODE;
static int __VERBOSE, __NO_LINK, __NO_CL;
static int __PREPROCESS_ONLY, __SHOW_INCLUDES;

static char *__concat_base (const char *str, ...) {
  va_list ap;
  char _buf[BIG_BUFFER];

  va_start (ap, str);
  _buf[0] = 0;
  
  while (str) {
    strcat_s (_buf, str);
    str = va_arg (ap, const char *);
  }
  
  va_end (ap);
  return _strdup (_buf);
}

static char *__concat (const char *s1) {
  return __concat_base (s1, 0);
}
static char *__concat (const char *s1, const char *s2) {
  return __concat_base (s1, s2, 0);
}
static char *__concat (const char *s1, const char *s2, const char *s3) {
  return __concat_base (s1, s2, s3, 0);
}
static char *__concat (const char *s1, const char *s2, const char *s3, const char *s4) {
  return __concat_base (s1, s2, s3, s4, 0);
}
static char *__concat (const char *s1, const char *s2, const char *s3, const char *s4, const char *s5) {
  return __concat_base (s1, s2, s3, s4, s5, 0);
}
static char *__concat (const char *s1, const char *s2, const char *s3, const char *s4, const char *s5, const char *s6) {
  return __concat_base (s1, s2, s3, s4, s5, s6, 0);
}
static char *__concat (const char *s1, const char *s2, const char *s3, const char *s4, const char *s5, const char *s6, const char *s7) {
  return __concat_base (s1, s2, s3, s4, s5, s6, s7, 0);
}
static char *__concat (const char *s1, const char *s2, const char *s3, const char *s4, const char *s5, const char *s6, const char *s7, const char *s8) {
  return __concat_base (s1, s2, s3, s4, s5, s6, s7, s8, 0);
}
  
static char *__basename (const char *f) {
  char target_drive[_MAX_DRIVE], target_dir[_MAX_DIR], target_fname[_MAX_FNAME], target_ext[_MAX_EXT];
  char _buf[_MAX_FNAME + _MAX_EXT + 1];
  if (_splitpath_s (f, target_drive, target_dir, target_fname, target_ext)) {
    return 0;
  }
  _snprintf_s (_buf, _MAX_FNAME + _MAX_EXT, "%s%s", target_fname, target_ext);
  return _strdup (_buf);
}

static char *__dirname (const char *f) {
  char target_drive[_MAX_DRIVE], target_dir[_MAX_DIR], target_fname[_MAX_FNAME], target_ext[_MAX_EXT];
  char _buf[_MAX_DRIVE + _MAX_DIR + 1];
  if (_splitpath_s (f, target_drive, target_dir, target_fname, target_ext)) {
    return 0;
  }
  _snprintf_s (_buf, _MAX_DRIVE + _MAX_DIR, "%s%s", target_drive, target_dir);
  size_t sz = strlen (_buf);
  if (strchr ("/\\", _buf[sz - 1])) {
    _buf[sz - 1] = 0;
  }  
  return _strdup (_buf);
}

static char *__chop_off_ext (const char *f) {
  char target_drive[_MAX_DRIVE], target_dir[_MAX_DIR], target_fname[_MAX_FNAME], target_ext[_MAX_EXT];
  char _buf[_MAX_DRIVE + _MAX_DIR + _MAX_FNAME + 1];
  if (_splitpath_s (f, target_drive, target_dir, target_fname, target_ext)) {
    return 0;
  }
  _snprintf_s (_buf, _MAX_DRIVE + _MAX_DIR + _MAX_FNAME, "%s%s%s", target_drive, target_dir, target_fname);
  return _strdup (_buf);
}

static char *__preserve_quotes (const char *f) {
  char _buf[BIG_BUFFER];
  char *p = _buf;
  while (*f) {
    /* we need to escape double quotes, and any backslashes that precede them */
    if (f[0] == '\"' || (f[0] == '\\' && f[1] == '\"')) {
      *p++ = '\\';
    }
    *p++ = *f++;
  }
  *p = 0;
  return _strdup (_buf);
}

static char *__replace_char (const char *s, char p, char q) {
  char _buf[BIG_BUFFER];
  char *b = _buf;
  do {
    *b = (*s == p? q: *s);
  } while (*s++, *b++);
  return _strdup (_buf);  
}

static char *__retrieve_line (const char **buf) {
  if (!*buf || !**buf) {
    return 0;
  }
  const char *end = *buf + strcspn (*buf, "\r\n");
  char _buf[BIG_BUFFER];
  strncpy_s (_buf, *buf, end - *buf);
  *buf = end;
  while (**buf && strchr ("\r\n", **buf)) {
    ++*buf;
  }
  return _strdup (_buf);
}

static char *__realpath (const char *f) {
  char *p = _fullpath (NULL, f, 0);
  char *q = p + strlen(p) - 1;
  if (q > p && strchr ("/\\", *q)) {
    *q = 0;
  }
  p[0] = toupper (p[0]);  // drive letter
  return __replace_char (p, '/', '\\');  
}

static int __starts_with (const char *s, const char *t) {
  return strncmp (s, t, strlen (t)) == 0;
}

static int __ends_with (const char *s, const char *t) {
  return strcmp (s + strlen (s) - strlen (t), t) == 0;
}
  
static int __matches (const char *s, const char *t) {
  return strcmp (s, t) == 0;
}

static char *__chop_off_last (const char *s, char c) {
  char _buf[_MAX_PATH + 1];
  strncpy_s (_buf, s, strrchr (s, c) - s);
  return _strdup (_buf);
}  

static int __file_exists (const char *F) {
  struct _stat _buf;
  
  if (_stat (F, &_buf) || !(_buf.st_mode & _S_IFREG)) {
    return 0;
  }
  return 1;
}

static int __empty_string (const char *s) {
  return !s || !*s;
}

static char *__read_file (const char *n) {
  char _buf[BIG_BUFFER + 1];
  FILE *f;
  fopen_s (&f, n, "r");
  size_t sz = fread (_buf, 1, BIG_BUFFER, f);
  fclose (f);
  _buf[sz] = 0;
  
  return _strdup (_buf);
}
  
static void __write_file (const char *n, const char *str) {
  FILE *f;
  fopen_s (&f, n, "w");
  fwrite (str, 1, strlen (str), f);
  fclose (f);
}

static void __append_file (const char *n, const char *str) {
  FILE *f;
  fopen_s (&f, n, "a+");
  fwrite (str, 1, strlen (str), f);
  fclose (f);
}

static void __touch_file (const char *n) {
  FILE *f;
  if (__VERBOSE) {
    fprintf (stderr, "%s --->> touch %s\n", SCRIPT, n);
  }
  fopen_s (&f, n, "a");
  fclose (f);
}
  
static char *__regtool_get (HKEY hive, const char *key, const char *sz_val) {
  char _buf[_MAX_PATH + 1];
  HKEY hKey;
  DWORD dwType=REG_SZ, dwSize=255;
  LONG returnStatus = RegOpenKeyEx (hive, key, 0L,  KEY_ALL_ACCESS, &hKey);
  _buf[0] = 0;
  if (returnStatus == ERROR_SUCCESS) {
     returnStatus = RegQueryValueEx (hKey, sz_val, NULL, &dwType,(LPBYTE)&_buf, &dwSize);
  }   
  RegCloseKey(hKey);
  return _strdup (_buf);   
}

static int __common (char ***argv) {
  char *_buf, _buf2[_MAX_PATH + 1];
  SCRIPT = __basename ((*argv)[0]);
  if (!(*argv)[1]) {
   show_error:
    fprintf (stderr, "%s: INVALID INVOCATION\n", SCRIPT);
    return 1;
  }
  
  __THIS_DIR = __realpath (__concat (__dirname ((*argv)[0]), "/.."));
  if (__starts_with ((*argv)[1], "--msvc-root=")) {
    __MSVC_ROOT = __concat ((*argv)[1] + 12);
    ++*argv;
    if (!(*argv)[1]) {
      goto show_error;
    }  
  } else {
    __MSVC_ROOT = __regtool_get (HKEY_LOCAL_MACHINE, "Software\\Microsoft\\VisualStudio\\8.0\\Setup\\VS", "ProductDir");
  }
  __MSVC_ROOT = __realpath (__MSVC_ROOT);
 
  _dupenv_s (&_buf, 0, "MSVC_VERBOSE"); 
  __VERBOSE = (_buf ? 1: 0);
  free (_buf);
  _dupenv_s (&_buf, 0, "PATH"); 
  PATH = _buf;
  
  GetSystemDirectory (_buf2, _MAX_PATH);
  SYSDIR = _strdup (_buf2);
  GetWindowsDirectory (_buf2, _MAX_PATH);
  SYSTEMROOT = _strdup (_buf2);
  
  __CL = __concat (__MSVC_ROOT, "\\VC\\bin\\cl.exe");
  __LIB = __concat (__MSVC_ROOT, "\\VC\\bin\\lib.exe");  
  __LINK = __concat (__MSVC_ROOT, "\\VC\\bin\\link.exe");  
  __BSCMAKE = __concat (__MSVC_ROOT, "\\VC\\bin\\bscmake.exe");  
  
  if (!__file_exists (__CL) || !__file_exists (__LIB) || !__file_exists (__LINK) || !__file_exists (__BSCMAKE)) {
    fprintf (stderr, "%s: CANNOT FIND MICROSOFT EXECUTABLES\n", SCRIPT);
    return 1;
  }
  
  INCLUDE = __concat (__THIS_DIR, "\\include;", __MSVC_ROOT, "\\VC\\include;", __MSVC_ROOT, "\\VC\\PlatformSDK\\Include;",
                      __MSVC_ROOT, "\\VC");
  LIB = __concat (__THIS_DIR, "\\lib;", __MSVC_ROOT, "\\VC\\lib");
  PATH = __concat (__MSVC_ROOT, "\\Common7\\IDE;", SYSDIR, ";", PATH);
  
  return 0;  
}

static int __invoke_cmd (char *__CMD) {
  STARTUPINFO si;
  PROCESS_INFORMATION pi;
  SECURITY_ATTRIBUTES sa;
  DWORD status;
  char _buf[BIG_BUFFER];
  char *_bufptr = _buf;

  if (__VERBOSE) {
    fprintf (stderr, "%s --->> %s\n", SCRIPT, __CMD);
  }
  
  _flushall ();

  ZeroMemory (&sa, sizeof (sa));
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = NULL;
  sa.bInheritHandle = TRUE;

  ZeroMemory (&si, sizeof (si));
  si.cb = sizeof (si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = GetStdHandle (STD_INPUT_HANDLE);
  if (__STDOUT_FILE) {
    si.hStdOutput = CreateFile (__STDOUT_FILE, GENERIC_READ | GENERIC_WRITE,
                               0, &sa, CREATE_ALWAYS, 
                               FILE_ATTRIBUTE_NORMAL,
                               NULL); 
    if (si.hStdOutput == INVALID_HANDLE_VALUE) {
     stdout_error:
      if (__VERBOSE) {
        fprintf (stderr, "%s: FATAL: Could not redirect stdout\n", SCRIPT);
      }
      return -1;
    }
  } else if (__DEPENDENCY_MODE) {
    char fname[16];
    tmpnam_s (fname, 15);
    // NB: A Race condition is still possible here...
    si.hStdOutput = CreateFile (fname, GENERIC_READ | GENERIC_WRITE,
                               0, &sa, CREATE_ALWAYS, 
                               FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
                               NULL); 
    if (si.hStdOutput == INVALID_HANDLE_VALUE) {
      goto stdout_error;
    }
  } else {
    si.hStdOutput = GetStdHandle (STD_OUTPUT_HANDLE);
  }
  si.hStdError = GetStdHandle (STD_ERROR_HANDLE);
  
  ZeroMemory (&pi, sizeof (pi));

  _bufptr += _snprintf_s (_bufptr, BIG_BUFFER / 4, _TRUNCATE, "INCLUDE=%s", INCLUDE);
  ++_bufptr;
  _bufptr += _snprintf_s (_bufptr, BIG_BUFFER / 4, _TRUNCATE, "LIB=%s", LIB);
  ++_bufptr;
  _bufptr += _snprintf_s (_bufptr, BIG_BUFFER / 4, _TRUNCATE, "PATH=%s", PATH);
  *++_bufptr;
  _bufptr += _snprintf_s (_bufptr, BIG_BUFFER / 4, _TRUNCATE, "SYSTEMROOT=%s", SYSTEMROOT);
  *++_bufptr = 0;
  
  // Start the child process. 
  if (!CreateProcess (NULL, __CMD, NULL, NULL, TRUE, 0, (LPVOID)_buf, NULL, &si, &pi)) {
    if (__VERBOSE) {
      fprintf (stderr, "%s: FATAL: Could not launch subprocess\n", SCRIPT);
    }
    return -1;
  }

  // Wait until child process exits.
  WaitForSingleObject (pi.hProcess, INFINITE);
  GetExitCodeProcess (pi.hProcess, &status);
  
  // Close process and thread handles. 
  CloseHandle (pi.hProcess);
  CloseHandle (pi.hThread);

  if (__STDOUT_FILE) {
    CloseHandle (si.hStdOutput);
  } else if (__DEPENDENCY_MODE) {
    DWORD zero = 0, written;
    WriteFile (si.hStdOutput, &zero, sizeof (zero), &written, NULL);
    HANDLE map = CreateFileMapping (si.hStdOutput, &sa, PAGE_READWRITE, 0, 0, NULL);
    if (!map) {
      goto map_error;
    }
    __STDOUT_MAP = (char *)MapViewOfFile (map, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!__STDOUT_MAP) {
     map_error:
      if (__VERBOSE) {
        fprintf (stderr, "%s: FATAL: Could not map stderr redirect\n", SCRIPT);
      }
      return -1;
    }
  }
     
  return status;
}

static int __build_browse_info (void) {
  __CMD = __concat (__BSCMAKE, " ", __BSCMAKE_VERSION, " ", __BSCMAKE_ARGS);
  char *bsc = __chop_off_last (__OUT, '.');
  __CMD = __concat (__CMD, " /n /Es /o ", bsc, ".bsc");
  bsc = __concat (bsc, ".bsc.cmd");
  __write_file (bsc, __BROWSE_FILES);
  __CMD = __concat (__CMD, " @", bsc);
  return __invoke_cmd (__CMD);
}