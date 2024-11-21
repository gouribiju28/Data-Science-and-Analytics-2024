## DATABASES

- Databases are sorted, structured form of data.
  * .csv files, .txt files, excel filesa re all called flat files
  * Data analysts retrieve the data stored in the database.
  * A database stores information in multiple locations to imporve reliability.
  * Examples of databases include Oracle, MySQL etc.
  * **Relational Database** are structured data with clear relationship between data points.
    + They have complex queries
    + It is usually in tabular form with rows and columns
  * **Non-Relational Databases** are less structured
    + They handle diverse data with rapid scalability needs
    + Includes graphs and documents.
    + Example : MongoDB
- A database contains separate tables for each category.
- The common connecting factor of each of the elements in the database is called *Primary Key*.
- All the data in a database are stored across multiple tables so as to avoid data duplication.

## SQL 

- Structured Query Language
- It is simple, powerful and globally used by data scientists all over the world.
- It is a domain specific language, unlike python which is general.
- It is a declarative programming language; the query does not say how to retrieve data from the dtabase, only which data to retrieve.
- There are four operations, namely : *CREATE*, *READ*, *UPDATE*, *DELETE*

- ### EXECUTION OF AN SQL STATEMENT
  + *Parser* is the part which understands the query.
  + *Query Optimizer* calculates how to get the query executed in the shortest time possible.
  + *Query Executer* executes the query.
 
- Schema refers to the tables and their relationship in the database.

### IMPORTANT QUERIES

SELECT <table_name>
FROM <row/colummn_name>
WHERE <specify any condition>
- WHERE query is used to filter.
OR : this query includes output where both the conditions are satisfied or if either of the condition is satisfied.
XOR : this query includes only one of the condition, not both at the same time.
LIKE
ROUND
DROP : deletes both contents and table.
TRUNCATE : deletes contents, but retains the table.

- SYNTAX:
  * SELECT * FROM<table> WHERE
    + GROUP BY
    + HAVING
    + ORDER BY (ASC, DESC)
    + LIMIT (OFFSET)
   
- JOIN : query used to merge two tables
  * INNER JOIN : only the common elements will be chosen and merged.
  * LEFT JOIN : starting from the left table, all elements will be included, null values used to represent data where it is not present in the right table.
  * RIGHT JOIN 
  * FULL OUTER JOIN : all the elements included
  * NATURAL JOIN
   
- NESTED QUERIES
  * Inner query will be executed first.
