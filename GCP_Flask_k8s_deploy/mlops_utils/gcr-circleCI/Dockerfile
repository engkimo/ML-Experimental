FROM golang:1.11-alpine
WORKDIR /go/src/github.com/naoty/go-web-server
COPY . .
RUN go install github.com/naoty/go-web-server
CMD ["go-web-server"]
 
