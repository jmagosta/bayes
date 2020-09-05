A visualization of the Law of Large Numbers

Intended to enlighten beginning statistics students. The Law of Large Numbers demonstrates how draws from any bounded population when averaged, converges to a normal distribution.  Understanding statistical sampling needs to know this.  This notebook shows how varying the size of the averages and the number of averages sampled affect convergence.  

To run this demonstration:

To run remotely (Use your local IP as the origin address):

    > bokeh serve LLN.py --allow-websocket-origin=192.168.15.100:5006

The remote client runs this in the browser as
http://192.168.15.100:5006

To run locally just invoke

    > bokeh serve LLN.py

To kill the process in powershell try cntl-del

