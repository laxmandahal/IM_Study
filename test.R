# Example of multivariate GAM

pdf("test.pdf")
N <- 200
I <- exp(rnorm(N, mean=0, sd=0.5))
mag <- rnorm(N, mean=3, sd=0.5)
log_acc <- I + rnorm(N, mean=1, sd=0.2)
log_disp <- 2*I + rnorm(N, mean=1, sd=0.5) / mag

df <- data.frame(log_acc=log_acc, log_disp=log_disp, I=I, mag=mag)

library(mgcv)

fit <- gam(list(log_acc  ~ I + s(mag),
                log_disp ~ I + s(mag)),
         data = df, family = mvn(d = 2))
summary(fit)
plot(fit)
#
# estimated cov matrix between log_acc and log_disp
cov <- solve(crossprod(fit$family$data$R))
cov
# correlation matrix between log_acc and log_disp
cov2cor(cov)
