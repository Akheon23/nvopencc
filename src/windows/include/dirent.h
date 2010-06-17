/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_DIRENT_H
#define __MINGW32_MSVC_DIRENT_H

#include "include/io.h"
#include "include/string.h"
#include "include/malloc.h"

#ifndef _MAX_PATH
#define _MAX_PATH 260
#endif

struct dirent {
  char d_name[_MAX_PATH];
};

typedef struct {
  intptr_t handle;
  char valid_entry;
  struct dirent entry;
} DIR;

DIR *opendir(const char *dirname) {
  struct _finddata_t fileinfo;
  intptr_t h = _findfirst(dirname, &fileinfo);
  DIR *dirp = 0;
  if(h != -1) {
    dirp = (DIR *)malloc(sizeof(DIR));
    dirp->handle = h;
    strncpy(dirp->entry.d_name, fileinfo.name, _MAX_PATH);
    dirp->valid_entry = 1;
  }
  return dirp;
}

struct dirent *readdir(DIR *dirp) {
  struct _finddata_t fileinfo;
  if(dirp->valid_entry) {
    dirp->valid_entry = 0;
    return &(dirp->entry);
  }
  if(_findnext(dirp->handle, &fileinfo)) {
    return 0;
  }
  strncpy(dirp->entry.d_name, fileinfo.name, _MAX_PATH);
  return &(dirp->entry);
}

int closedir(DIR *dirp) {
  _findclose(dirp->handle);
  free(dirp);
  return 0;
}

#endif // #ifndef __MINGW32_MSVC_DIRENT_H
