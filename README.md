# Microscope Toolset
This repository is a toolset for microscope that use pymmcore-plus with LLM


### How to get  started

In order to use this toolset, you need to have python installed. There are differents option, you could use *Anaconda*, *miniconda* or *mamba*. After you installed your favorite package and environment management system create a specific environment for this toolset.

```
conda create -n microscope-toolset python=3.12.11
```
Afterwards activate your newly environment
```
conda activate microscope-toolset
```
Create a new folder and clone the _microscope-toolset_ repository
```
git clone https://github.com/ddd42-star/microscope-toolset.git
```
Then go into the folder of this repository and install all the packages using the *requirements* file
```
pip install -r requirements.txt
```

#### Postgresql
To improve our agents, we created a *Postgresql* database that will collect all the results obtained by our differents tool. To use it, you will need to download [Postgressql](https://www.postgresql.org/download/) based on your OS.

*Window installation*

There are plenty of tutorial for the installation available online, just follow the most appropriate one. After the installation go in your terminal and digit the following:
```
 > psql -U postgres -p 5410
```
*-U* is the username of the user. The default one is **postgres** and is the one with the most privilige. *-p* is to specify the port number: if during the installation, you changed the port number from **5432**(the default one) to another, specify the correct one. After this, you will be asked to insert the password for the user *postgres*
```
Insert the password for the user postgres:
```
Here you need to put the password you used as a *master password*.
Finally, you will see the psql console:
```
postgres=# ...
```
Now before continuing, you need to:

1. Create a new *user* with a new *password*
2. Create a new *database* owned by the new user

In this way the data will be saved into a 'local' user and not the 'admin' one called postgres.
```
> postgres=# CREATE USER <new_user> WITH PASSWORD '<newly_password>';
> CREATE ROLE
> postgres=# CREATE DATABASE <database name> OWNER <new_user>;      
> CREATE DATABASE
```
After this you can quit
```
> postgres=# \q
```
and save the new information into the *.env* file.

 - DB_NAME=<_The name of the database_>
 - DB_USER=<_The new user name created_>
 - DB_PASSWORD=<_The new password of the new user_>

*DB_HOST* still remains _localhost_ and *DB_PORT* still remains the one you choosed during the installation. If an error occured, you should not be able to connect at your database and you will not able to start the GUI.

### Create the vector database with pdf files
To help the different agents to avoid hallucination, it's advised to create a vector database with the different "knowledge". We have the documentation of *pymmcore_plus* and the publications of the *Pertz Lab*. If you want to add other pdfs file you can run this command:
```
python .\src\create_database_from_publication.py --db <path to db> --doc <path to pdf(s)>
```
In the folder that you choose were to save the database, at the moment two directories will be created: *pages_png* and *pages_markdown*. The first will contain the png files of each page of the document and in the second the markdown files of the text extracted. 

### How to start the toolset

> Currently, the *Large Language Model* in use is from OpenAI. In the future it will be added the possibilities to choose the preferred Model. Be sure to have saved in the environment the api key for OpenAI with *OPENAI_API_KEY*.

Run the following command for starting the Napari GUI
```
python -m src.plugin_napari
```
On the right there is the panel control that will start or stop the MCP Microscope Toolset server.

In the UI at your choice (e.g. vs-code, Claude Desktop) add the configuration file of the mcp server.
```
Example of vs-code
{
	"servers": {
		"microscope": {
			"url": "http://127.0.0.1:5500/mcp",
			"type": "http"
		}
	}
}
```
After you added the *mcp.json* configuration file, you can start the MCP Client that will connect to the server.

```
Example of claude code

$ claude code add --transport http microscope http://127.0.0.1:5500/mcp

Then goes in /mcp

$ /mcp + enter

And select the microscope MCP server either connecting the server or enabling the server, and from the terminal whery you started the napari-plugin you will see if the server correctly connected.
```

