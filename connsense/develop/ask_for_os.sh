  # call as ./ask_for_os.sh NAME, where NAME is who to ask

  if [ -z "$1" ]; then
      asked="$USER"
  else
      asked="$1"
  fi

  echo Hi, "$asked"! What operating system are you using?
  read my_os

  if [ "$asked" = "RMS" ]; then
      echo You\'re using GNU/"$my_os"!
  elif [ "$asked" = "Linus" ]; then
      echo You\'re using "$my_os"!
  else
      echo You\'re using `uname -o`!
  fi
