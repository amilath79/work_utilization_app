{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9acae4ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import pyodbc\n",
    "import os\n",
    "from datetime import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0c964336",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SQL Server connection parameters\n",
    "server = 'your_server_name'  \n",
    "database = 'your_database_name'\n",
    "username = 'your_username'\n",
    "password = 'your_password'\n",
    "driver = '{ODBC Driver 17 for SQL Server}'  # Update with your driver version\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b409a66e",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'os' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mNameError\u001b[39m                                 Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[3]\u001b[39m\u001b[32m, line 3\u001b[39m\n\u001b[32m      1\u001b[39m \u001b[38;5;66;03m# Create output directory if it doesn't exist\u001b[39;00m\n\u001b[32m      2\u001b[39m output_dir = \u001b[33m\"\u001b[39m\u001b[33mworkforce_data\u001b[39m\u001b[33m\"\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m3\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[43mos\u001b[49m.path.exists(output_dir):\n\u001b[32m      4\u001b[39m     os.makedirs(output_dir)\n",
      "\u001b[31mNameError\u001b[39m: name 'os' is not defined"
     ]
    }
   ],
   "source": [
    "# Create output directory if it doesn't exist\n",
    "output_dir = \"workforce_data\"\n",
    "if not os.path.exists(output_dir):\n",
    "    os.makedirs(output_dir)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
